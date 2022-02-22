module MLJGLMInterface

# -------------------------------------------------------------------
# TODO
# - return feature importance curve to report using `features`
# - handle binomial case properly, needs MLJ API change for weighted
# samples (y/N ~ Be(p) with weights N)
# - handle levels properly (see GLM.jl/issues/240); if feed something
# with levels, the fit will fail.
# - revisit and test Poisson and Negbin regression once there's a clear
# example we can test on (requires handling levels which deps upon GLM)
# - test Logit, Probit etc on Binomial once binomial case is handled
# -------------------------------------------------------------------

export LinearRegressor, LinearBinaryClassifier, LinearCountRegressor

import MLJModelInterface
import MLJModelInterface: metadata_pkg, metadata_model, Table, Continuous, Count, Finite,
    OrderedFactor, Multiclass, @mlj_model
using Distributions: Bernoulli, Distribution, Poisson
using Tables
import GLM
using GLM: FormulaTerm

const MMI = MLJModelInterface
const PKG = "MLJGLMInterface"

##
## DESCRIPTIONS
##

const LR_DESCR = "Linear regressor (OLS) with a Normal model."
const LBC_DESCR = "Linear binary classifier with "*
    "specified link (e.g. logistic)."
const LCR_DESCR = "Linear count regressor with specified "*
    "link and distribution (e.g. log link and poisson)."


####
#### REGRESSION TYPES
####

# LinearRegressor        --> Probabilistic w Continuous Target
# LinearCountRegressor   --> Probabilistic w Count Target
# LinearBinaryClassifier --> Probabilistic w Binary target // logit,cauchit,..
# MulticlassClassifier   --> Probabilistic w Multiclass target

const VALID_KEYS = [:deviance, :dof_residual, :stderror, :vcov, :coef_table]
const DEFAULT_KEYS = VALID_KEYS # For more understandable warning mssg by `@mlj_model`.
const KEYS_TYPE = Union{Nothing, AbstractVector{Symbol}}

@mlj_model mutable struct LinearRegressor <: MMI.Probabilistic
    fit_intercept::Bool = true
    dropcollinear::Bool = false
    offsetcol::Union{Symbol, Nothing} = nothing
    report_keys::KEYS_TYPE = DEFAULT_KEYS::(isnothing(_) || issubset(_, VALID_KEYS))
end

@mlj_model mutable struct LinearBinaryClassifier <: MMI.Probabilistic
    fit_intercept::Bool = true
    link::GLM.Link01 = GLM.LogitLink()
    offsetcol::Union{Symbol, Nothing} = nothing
    maxiter::Integer = 30
    atol::Real = 1e-6
    rtol::Real = 1e-6
    minstepfac::Real = 0.001
    report_keys::KEYS_TYPE = DEFAULT_KEYS::(isnothing(_) || issubset(_, VALID_KEYS))
end

@mlj_model mutable struct LinearCountRegressor <: MMI.Probabilistic
    fit_intercept::Bool = true
    distribution::Distribution = Poisson()
    link::GLM.Link = GLM.LogLink()
    offsetcol::Union{Symbol, Nothing} = nothing
    maxiter::Integer = 30
    atol::Real = 1e-6
    rtol::Real = 1e-6
    minstepfac::Real = 0.001
    report_keys::KEYS_TYPE = DEFAULT_KEYS::(isnothing(_) || issubset(_, VALID_KEYS))
end

# Short names for convenience here

const GLM_MODELS = Union{
    <:LinearRegressor, <:LinearBinaryClassifier, <:LinearCountRegressor
}

###
## Helper functions
###

"""
augment_X(X, b)
Augment the matrix `X` with a column of ones if the intercept is to be
fitted (`b=true`), return `X` otherwise.
"""
function augment_X(X::Matrix, b::Bool)::Matrix
    b && return hcat(X, ones(float(Int), size(X, 1), 1))
    return X
end

_to_vector(v::Vector) = v
_to_vector(v) = collect(v)
_to_array(v::AbstractArray) = v
_to_array(v) = collect(v)

"""
    split_X_offset(X, offsetcol::Nothing)

When no offset is specified, return `X` and an empty vector.
"""
split_X_offset(X, offsetcol::Nothing) = (X, Float64[])

"""
    split_X_offset(X, offsetcol::Symbol)

Splits the input table X in:
    - A new table not containing the original offset column
    - The offset vector extracted from the table
"""
function split_X_offset(X, offsetcol::Symbol)
    ct = Tables.columntable(X)
    offset = Tables.getcolumn(ct, offsetcol)
    newX = Base.structdiff(ct, NamedTuple{(offsetcol,)})
    return newX, _to_vector(offset)
end

# If `estimates_dispersion_param` returns `false` then the dispersion
# parameter isn't estimated from data but known apriori to be `1`.
estimates_dispersion_param(::LinearRegressor) = true
estimates_dispersion_param(::LinearBinaryClassifier) = false

function estimates_dispersion_param(model::LinearCountRegressor)
    return GLM.dispersion_parameter(model.distribution)
end

function _throw_sample_size_error(model, est_dispersion_param)
    requires_info = _requires_info(model, est_dispersion_param)

    if isnothing(model.offsetcol)
        offset_info = " `offsetcol == nothing`"
    else
        offset_info = " `offsetcol !== nothing`"
    end

    modelname = nameof(typeof(model))
    if model isa LinearCountRegressor    
        distribution_info = "and `distribution = $(nameof(typeof(model.distribution)))()`"
    else
        distribution_info = "\b"
    end

    throw(
        ArgumentError(
            " `$(modelname)` with `fit_intercept = $(model.fit_intercept)`,"*
            "$(offset_info) $(distribution_info) requires $(requires_info)"
        )
    )
    return nothing
end

""" 
    _requires_info(model, est_dispersion_param)
    
Returns one of the following strings
- "`n_samples >= n_features`", "`n_samples > n_features`"
- "`n_samples >= n_features - 1`",  "`n_samples > n_features - 1`"
- "`n_samples >= n_features + 1`", "`n_samples > n_features + 1`"
"""
function _requires_info(model, est_dispersion_param)
    inequality = est_dispersion_param ? ">" : ">="
    int_num = model.fit_intercept - !isnothing(model.offsetcol)

    if iszero(int_num)
        int_num_string = "\b"
    elseif int_num < 0
        int_num_string = "- $(abs(int_num))"
    else
        int_num_string = "+ $(int_num)"
    end

    return "`n_samples $(inequality) n_features $(int_num_string)`."
end

function check_sample_size(model, n, p)
    if estimates_dispersion_param(model)
        n <= p + model.fit_intercept && _throw_sample_size_error(model, true)
    else
        n < p + model.fit_intercept && _throw_sample_size_error(model, false)
    end
    return nothing
end

function _matrix_and_features(model, Xcols, handle_intercept=false)
    col_names = Tables.columnnames(Xcols)
    n, p = Tables.rowcount(Xcols), length(col_names)
    augment = handle_intercept && model.fit_intercept

    if !handle_intercept # i.e This only runs during `fit`
        check_sample_size(model, n, p)
    end

    if p == 0
        Xmatrix = Matrix{float(Int)}(undef, n, p)
    else
        Xmatrix = Tables.matrix(Xcols)
    end

    Xmatrix = augment_X(Xmatrix, augment)

    return Xmatrix, col_names
end

_to_columns(t::Tables.AbstractColumns) = t
_to_columns(t) = Tables.Columns(t)

"""
    prepare_inputs(model, X; handle_intercept=false)

Handle `model.offsetcol` and `model.fit_intercept` if `handle_intercept=true`.
`handle_intercept` is disabled for fitting since the StatsModels.@formula handles the intercept.
"""
function prepare_inputs(model, X; handle_intercept=false)
    Xcols = _to_columns(X)
    table_features = Tables.columnnames(Xcols)
    p = length(table_features)
    p >= 1 || throw(
        ArgumentError("`X` must contain at least one feature column.")
    )
    if !isnothing(model.offsetcol)
        model.offsetcol in table_features || throw(
            ArgumentError("offset column `$(model.offsetcol)` not found in table `X")
        )
        if p < 2 && !model.fit_intercept
            throw(
                ArgumentError(
                    "At least 2 feature columns are required for learning with"*
                    " `offsetcol !== nothing` and `fit_intercept == false`."
                )
            )
        end
    end
    Xminoffset, offset = split_X_offset(Xcols, model.offsetcol)
    Xminoffset_cols = _to_columns(Xminoffset)
    Xmatrix, features = _matrix_and_features(model, Xminoffset_cols , handle_intercept)
    return Xmatrix, offset, _to_array(features)
end

"""
    glm_report(glm_model, features, reportkeys)

Report based on a fitted `LinearModel/GeneralizedLinearModel`, `glm_model` 
and keyed on `reportkeys`.
"""
glm_report

glm_report(glm_model, features, ::Nothing) = NamedTuple()

function glm_report(glm_model, features, reportkeys)
    isempty(reportkeys) && return NamedTuple()
    report_dict = Dict{Symbol, Any}() 
    if in(:deviance, reportkeys)
        report_dict[:deviance] = GLM.deviance(glm_model)
    end
    if in(:dof_residual, reportkeys)
        report_dict[:dof_residual] = GLM.dof_residual(glm_model)
    end
    if in(:stderror, reportkeys)
        report_dict[:stderror] = GLM.stderror(glm_model)
    end
    if :vcov in reportkeys
        report_dict[:vcov] = GLM.vcov(glm_model)
    end
    if :coef_table in reportkeys
        coef_table = GLM.coeftable(glm_model)
        # Update the variable names in the `coef_table` with the actual variable
        # names seen during fit.
        if length(coef_table.rownms) == length(features)
            # This means `fit_intercept` is false
            coef_table.rownms = string.(features)
        else
            coef_table.rownms = [string.(features); "(Intercept)"]
        end
        report_dict[:coef_table] = coef_table
    end
    return NamedTuple{Tuple(keys(report_dict))}(values(report_dict))
end

"""
    glm_formula(model, features) -> FormulaTerm

Return formula which is ready to be passed to `fit(form, data, ...)`.
"""
function glm_formula(model, features)::FormulaTerm
    # By default, using a JuliaStats formula will add an intercept.
    # Adding a zero term explicitly disables the intercept.
    # See the StatsModels.jl tests for more information.
    intercept_term = model.fit_intercept ? 1 : 0
    form = GLM.Term(:y) ~ sum(GLM.term.(features)) + GLM.term(intercept_term)
    return form
end

"""
    glm_data(model, Xmatrix, y, features)

Return data which is ready to be passed to `fit(form, data, ...)`.
"""
function glm_data(model, Xmatrix, y, features)
    data = Tables.table([Xmatrix y]; header=[features...; :y])
    return data
end

"""
    check_weights(w, y)

Validate observation weights to be used in fitting `Linear/GeneralizedLinearModel` based 
on target vector `y`.
Note: 
Categorical targets have to be converted into a AbstractVector{<:Real} before 
passing to this method.
"""
check_weights

check_weights(::Nothing, y::AbstractVector{<:Real}) = similar(y, 0)

function check_weights(w, y::AbstractVector{<:Real})
    w isa AbstractVector{<:Real} || throw(
        ArgumentError("Expected `weights === nothing` or `weights::AbstractVector{<:Real}")
    )
    length(y) == length(w) || throw(
        ArgumentError("weights passed must have the same length as the target vector.")
    )
    return w
end


####
#### FIT FUNCTIONS
####

struct FitResult{V<:AbstractVector, T, R}
    "Vector containg coeficients of the predictors and intercept"
    coefs::V
    "An estimate of the dispersion parameter of the glm model. "
    dispersion::T
    "Other fitted parameters specific to a fitted model"
    params::R
end

coefs(fr::FitResult) = fr.coefs
dispersion(fr::FitResult) = fr.dispersion
params(fr::FitResult) = fr.params

function MMI.fit(model::LinearRegressor, verbosity::Int, X, y, w=nothing)
    # apply the model
    Xmatrix, offset, features = prepare_inputs(model, X)
    y_ = isempty(offset) ? y : y .- offset
    wts = check_weights(w, y_)
    data = glm_data(model, Xmatrix, y_, features)
    form = glm_formula(model, features)
    fitted_lm = GLM.lm(form, data; model.dropcollinear, wts).model
    fitresult = FitResult(
        GLM.coef(fitted_lm), GLM.dispersion(fitted_lm), (features = features,)
    )
    # form the report
    report = glm_report(fitted_lm, features, model.report_keys)
    cache = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearCountRegressor, verbosity::Int, X, y, w=nothing)
    # apply the model
    Xmatrix, offset, features = prepare_inputs(model, X)
    data = glm_data(model, Xmatrix, y, features)
    wts = check_weights(w, y)
    form = glm_formula(model, features)
    fitted_glm_frame = GLM.glm(
        form, data, model.distribution, model.link;
        offset,
        model.maxiter,
        model.atol,
        model.rtol,
        model.minstepfac,
        wts
    )
    fitted_glm = fitted_glm_frame.model
    fitresult = FitResult(
        GLM.coef(fitted_glm), GLM.dispersion(fitted_glm), (features = features,)
    )
    # form the report
    report = glm_report(fitted_glm, features, model.report_keys)
    cache = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearBinaryClassifier, verbosity::Int, X, y, w=nothing)
    # apply the model
    decode = y[1]
    y_plain = MMI.int(y) .- 1 # 0, 1 of type Int
    wts = check_weights(w, y_plain)
    Xmatrix, offset, features = prepare_inputs(model, X)
    data = glm_data(model, Xmatrix, y_plain, features)
    form = glm_formula(model, features)
    fitted_glm_frame = GLM.glm(
        form, data, Bernoulli(), model.link;
        offset,
        model.maxiter,
        model.atol,
        model.rtol,
        model.minstepfac,
        wts
    )
    fitted_glm = fitted_glm_frame.model
    fitresult = FitResult(
        GLM.coef(fitted_glm), GLM.dispersion(fitted_glm), (features = features,)
    )
    # form the report
    report = glm_report(fitted_glm, features, model.report_keys)
    cache = nothing
    # return
    return (fitresult, decode), cache, report
end

glm_fitresult(::LinearRegressor, fitresult) = fitresult
glm_fitresult(::LinearCountRegressor, fitresult) = fitresult
glm_fitresult(::LinearBinaryClassifier, fitresult) = fitresult[1]

function MMI.fitted_params(model::GLM_MODELS, fitresult)
    result = glm_fitresult(model, fitresult)
    coef = coefs(result)
    features = copy(params(result).features)
    if model.fit_intercept
        intercept = coef[end]
        coef_ = coef[1:end-1]
    else
        intercept = zero(eltype(coef))
        coef_ = copy(coef)
    end
    return (; features, coef=coef_, intercept)
end


####
#### PREDICT FUNCTIONS
####

glm_link(model) = model.link
glm_link(::LinearRegressor) = GLM.IdentityLink()

# more efficient than MLJBase fallback
function MMI.predict_mean(model::GLM_MODELS, fitresult, Xnew)
    Xmatrix, offset, _ = prepare_inputs(model, Xnew; handle_intercept=true)
    result = glm_fitresult(model, fitresult) # ::FitResult
    coef = coefs(result)
    p = size(Xmatrix, 2)
    if p != length(coef)
        throw(
            DimensionMismatch(
                "The number of features in training and prediction datasets must be equal"
            )
        )
    end
    link = glm_link(model)
    return glm_predict(link, coef, Xmatrix, model.offsetcol, offset)
end

# barrier function to aid performance
function glm_predict(link, coef, Xmatrix, offsetcol, offset) 
    η = offsetcol === nothing ? (Xmatrix * coef) : (Xmatrix * coef .+ offset)
    μ = GLM.linkinv.(link, η)
    return μ
end

function MMI.predict(model::LinearRegressor, fitresult, Xnew)
    μ = MMI.predict_mean(model, fitresult, Xnew)
    σ̂ = dispersion(fitresult)
    return [GLM.Normal(μᵢ, σ̂) for μᵢ ∈ μ]
end

function MMI.predict(model::LinearCountRegressor, fitresult, Xnew)
    λ = MMI.predict_mean(model, fitresult, Xnew)
    return [GLM.Poisson(λᵢ) for λᵢ ∈ λ]
end

function MMI.predict(model::LinearBinaryClassifier, (fitresult, decode), Xnew)
    π = MMI.predict_mean(model, (fitresult, decode), Xnew)
    return MMI.UnivariateFinite(MMI.classes(decode), π, augment=true)
end

# NOTE: predict_mode uses MLJBase's fallback

####
#### METADATA
####

# shared metadata
const GLM_REGS = Union{
    Type{<:LinearRegressor}, Type{<:LinearBinaryClassifier}, Type{<:LinearCountRegressor}
}

metadata_pkg.(
    (LinearRegressor, LinearBinaryClassifier, LinearCountRegressor),
    name = "GLM",
    uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a",
    url = "https://github.com/JuliaStats/GLM.jl",
    julia = true,
    license = "MIT",
    is_wrapper = false
)

metadata_model(
    LinearRegressor,
    input = Table(Continuous),
    target = AbstractVector{Continuous},
    supports_weights = true,
    descr = LR_DESCR,
    path = "$PKG.LinearRegressor"
)

metadata_model(
    LinearBinaryClassifier,
    input = Table(Continuous),
    target = AbstractVector{<:Finite{2}},
    supports_weights = true,
    descr = LBC_DESCR,
    path = "$PKG.LinearBinaryClassifier"
)

metadata_model(
    LinearCountRegressor,
    input = Table(Continuous),
    target = AbstractVector{Count},
    supports_weights = true,
    descr = LCR_DESCR,
    path = "$PKG.LinearCountRegressor"
)

end # module
