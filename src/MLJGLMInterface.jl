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
import MLJModelInterface: metadata_pkg, metadata_model,
    Table, Continuous, Count, Finite, OrderedFactor,
    Multiclass, @mlj_model
import Distributions
using Tables
import GLM

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

const VALID_REPORT_PARMS = [:deviance, :dof_residual, :stderror, :vcov, :coef_table]

MMI.@mlj_model mutable struct LinearRegressor <: MMI.Probabilistic
    fit_intercept::Bool = true
    dropcollinear::Bool = true
    offsetcol::Union{Symbol, Nothing} = nothing
    report_params::Union{
        Nothing, AbstractVector{Symbol}
    } = VALID_REPORT_PARMS::(_ === nothing || issubset(_, VALID_REPORT_PARMS))
    check_features::Bool = false
end

MMI.@mlj_model mutable struct LinearBinaryClassifier <: MMI.Probabilistic
    fit_intercept::Bool = true
    link::GLM.Link01 = GLM.LogitLink()
    offsetcol::Union{Symbol, Nothing} = nothing
    maxiter::Integer = 30
    atol::Real = 1e-6
    rtol::Real = 1e-6
    minstepfac::Real = 0.001
    report_params::Union{
        Nothing, AbstractVector{Symbol}
    } = VALID_REPORT_PARMS::(_ === nothing || issubset(_, VALID_REPORT_PARMS))
    check_features::Bool = false
end

MMI.@mlj_model mutable struct LinearCountRegressor <: MMI.Probabilistic
    fit_intercept::Bool = true
    distribution::Distributions.Distribution = Distributions.Poisson()
    link::GLM.Link = GLM.LogLink()
    offsetcol::Union{Symbol, Nothing} = nothing
    maxiter::Integer = 30
    atol::Real = 1e-6
    rtol::Real = 1e-6
    minstepfac::Real = 0.001
    report_params::Union{
        Nothing, AbstractVector{Symbol}
    } = VALID_REPORT_PARMS::(_ === nothing || issubset(_, VALID_REPORT_PARMS))
    check_features::Bool = false
end

# Short names for convenience here

const GLM_MODELS = Union{
    <:LinearRegressor, 
    <:LinearBinaryClassifier, 
    <:LinearCountRegressor
}


###
## Helper functions
###

function _throw_dims_err(model, est_dispersion_param)
    add_to = model.fit_intercept - !isnothing(model.offsetcol)
    throw(
        ArgumentError(
            " A `$(nameof(typeof(model)))` with `fit_intercept == $(model.fit_intercept)`,"
            * ifelse(
                isnothing(model.offsetcol),
                " `offsetcol == nothing` ",
                " `offsetcol !== nothing` "
            ) *
            ifelse(
                model isa LinearCountRegressor,   
                "and `distribution::$(nameof(typeof(model.distribution)))` ",
                ""
            ) *
            "require `n_samples $(ifelse(est_dispersion_param, ">", ">=")) "*
            "n_features $(
                ifelse(iszero(add_to), "", ifelse(add_to < 0, "-", "+") * "$(abs(add_to))")
            )`."
        )
    )
    return nothing
end

_getarray(iter::AbstractVector) = iter
_getarray(iter) = collect(iter)

# If `estimates_dispersion_param` returns `false` then the dispersion
# parameter isn't estimated from data but known apriori to be `1`.
estimates_dispersion_param(::LinearRegressor) = true
estimates_dispersion_param(::LinearBinaryClassifier) = false

function estimates_dispersion_param(model::LinearCountRegressor)
    return GLM.dispersion_parameter(model.distribution)
end
function check_dims(model, n, p)
    if estimates_dispersion_param(model)
        n <= p + model.fit_intercept && _throw_dims_err(model, true)
    else
        n < p + model.fit_intercept && _throw_dims_err(model, false)
    end
end

function _to_matrix(model, X::Tables.MatrixTable, check=true)
    sch = Tables.schema(X)
    n, p = Tables.rowcount(X), length(sch.names)
    if check
        check_dims(model, n, p)
    end

    if model.fit_intercept
        p = p + 1
        T = reduce(promote_type, sch.types)
        matrix = Matrix{T}(undef, n, p)
        for (i, col) in enumerate(X)
            matrix[:, i] .= col
        end
        (@inbounds(matrix[:, p] .= one(T)))
    else
        matrix = Tables.matrix(X)
    end
    return matrix
end

function _to_matrix(model, X, check=true)
    cols = Tables.Columns(X)
    types = Tables.schema(cols).types
    T = reduce(promote_type, types)
    n, p = Tables.rowcount(cols), length(types)
    if check
        check_dims(model, n, p)
    end
    
    matrix = Matrix{T}(undef, n, p + model.fit_intercept)
    for (i, col) in enumerate(cols)
        matrix[:, i] .= col
    end
    model.fit_intercept && (@inbounds(matrix[:, p + 1] .= one(T)))
    
    return matrix
end

"""
    split_X_offset(X, offsetcol::Nothing)

When no offset is specied, return `X`, an empty vector and its features.
"""
function split_X_offset(X, offsetcol::Nothing)
    
   return (X, Float64[], _getarray(Tables.columnnames(X)))
end

## TODO - find a way to split_X_offset without needing to convert `X` into a `ColumnTable`
"""
    split_X_offset(X, offsetcol::Symbol)

Splits the input X into:
    - A new table not containing the original offset column
    - The offset vector extracted from the table
Additional it also returns the features of the new table
"""
function split_X_offset(X, offsetcol::Symbol)
    Xcols = Tables.columntable(X)
    #sch = Tables.schema(Xcols)
    offset = Tables.getcolumn(Xcols, offsetcol)
    newX = Base.structdiff(Xcols, NamedTuple{(offsetcol,)})
    return newX, Vector(offset), _getarray(Tables.columnnames(newX))
end

"""
    prepare_inputs(model, X, check=false)

Handle `model.offsetcol` (if present) and `model.fit_intercept`.
"""
function prepare_inputs(model, X, check=true)
    Xcols = (X isa Tables.MatrixTable || X isa Tables.ColumnTable) ? X : Tables.Columns(X)
    table_features = Tables.columnnames(Xcols)
    length(table_features) >= 1 || throw(
        ArgumentError("`X` must contain at least one feature column.")
    )
    if !isnothing(model.offsetcol)
        model.offsetcol in table_features || throw(
            ArgumentError("offset column `$(model.offsetcol)` not found in table `X")
        )
        if length(table_features) < 2 && !model.fit_intercept
            throw(
                ArgumentError(
                    "At least 2 feature columns are required for learning with"*
                    " `offsetcol !== nothing` and `fit_intercept == false`."
                )
            )
        end
    end
    Xminoffset, offset, features = split_X_offset(Xcols, model.offsetcol)
    Xmatrix = _to_matrix(model, Xminoffset, check)
    return Xmatrix, offset, features
end

"""
    glm_report(glm_models, reportkeys, features)

Returns a Dictionary report based on a fitted `GeneralizedLinearModel` or `LinearModel` 
and keyed on `reportkeys`. The `features` argument is only useful when `:coef_table` is
one of the keys in `reportkeys`.
"""
glm_report

glm_report(glm_model, ::Nothing) = NamedTuple()

function glm_report(glm_model, reportkeys, ftrs)
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
    if in(:vcov, reportkeys)
        report_dict[:vcov] = GLM.vcov(glm_model)
    end
    if in(:coef_table, reportkeys)
        coef_table = GLM.coeftable(glm_model)
        if length(coef_table.rownms) == length(ftrs)
            # This means `fit_intercept` is false
            coef_table.rownms = string.(ftrs)
        else      
            coef_table.rownms = [
                (string(ftrs[i]) for i in eachindex(ftrs))...;
                "(Intercept)"
            ]
        end
        report_dict[:coef_table] = coef_table
    end
    return NamedTuple{Tuple(keys(report_dict))}(values(report_dict))
end


"""
    check_weights(w, y)

Validate sample weights to be used in fitting `Linear/GeneralizedLinearModel` based 
on target vector `y`.
Note: Categorical targets have to be converted into a AbstractVector{<:Real} before 
passing to this method.
"""
check_weights

check_weights(::Nothing, y::AbstractVector{<:Real}) = similar(y, 0)

function check_weights(w, y::AbstractVector{<:Real})
    w isa AbstractVector{<:Real} || throw(
        ArgumentError("weights passed must be an instance of `AbstractVector{<:Real}")
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
    coef::V
    "An estimate of the dispersion parameter of the glm model. "
    dispersion::T
    "Other fitted parameters specific to a fitted model"
    params::R
    function FitResult(
        coef::AbstractVector,
        dispersion,
        params
    )
        return new{
            typeof(coef),
            typeof(dispersion),
            typeof(params)
        }(coef, dispersion, params)
    end
end

coef(fr::FitResult) = fr.coef
dispersion(fr::FitResult) = fr.dispersion
params(fr::FitResult) = fr.params

function MMI.fit(model::LinearRegressor, verbosity::Int, X, y, w=nothing)
    # apply the model
    Xmatrix, offset, features = prepare_inputs(model, X)
    y_ = isnothing(model.offsetcol) ? y : y .- offset
    w_ = check_weights(w, y_)
    linear_reg_model = GLM.lm(Xmatrix, y_; wts= w_, dropcollinear=model.dropcollinear)
    fitresult = FitResult(
        GLM.coef(linear_reg_model),
        GLM.dispersion(linear_reg_model),
        (feature_names = features,)
    )
    # form the report
    report = glm_report(linear_reg_model, model.report_params, features)
    cache = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearCountRegressor, verbosity::Int, X, y, w=nothing)
    # apply the model
    w_ = check_weights(w, y)
    Xmatrix, offset, features = prepare_inputs(model, X)  
    glm_reg_model = GLM.glm(
        Xmatrix, y, model.distribution, model.link;
        wts = w_,
        offset=offset,
        maxiter=model.maxiter,
        atol = model.atol,
        rtol = model.rtol,
        minstepfac = model.minstepfac
    )

    fitresult = FitResult(
        GLM.coef(glm_reg_model),
        GLM.dispersion(glm_reg_model),
        (feature_names = features,)
    )
    # form the report
    report = glm_report(glm_reg_model, model.report_params, features)
    cache = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearBinaryClassifier, verbosity::Int, X, y, w=nothing)
    # apply the model
    decode = y[1]
    y_plain = MMI.int(y) .- 1 # 0, 1 of type Int
    w_ = check_weights(w, y_plain)
    Xmatrix, offset, features = prepare_inputs(model, X) 
    glm_reg_model = GLM.glm(
        Xmatrix, y_plain, Distributions.Bernoulli(), model.link;
        wts = w_,
        offset=offset,
        maxiter=model.maxiter,
        atol = model.atol,
        rtol = model.rtol,
        minstepfac = model.minstepfac
    )
    fitresult = FitResult(
        GLM.coef(glm_reg_model),
        GLM.dispersion(glm_reg_model),
        (feature_names = features,)
    )
    # form the report
    report = glm_report(glm_reg_model, model.report_params, features)
    cache = nothing
    # return
    return (fitresult, decode), cache, report
end

glm_fitresult(::Union{LinearRegressor, LinearCountRegressor}, fitresult) = fitresult
glm_fitresult(::LinearBinaryClassifier, fitresult) = fitresult[1]

function MMI.fitted_params(model::GLM_MODELS, fitresult)
    result = glm_fitresult(model, fitresult)
    coef_ = coef(result)
    feature_names = copy(params(result).feature_names)
    if model.fit_intercept
        intercept = coef_[end]
        coef_ = coef_[1:end-1]
    else
        intercept = zero(eltype(coef_))
        coef_ = copy(coef_)
    end
    return (; feature_names=feature_names, coef=coef_, intercept=intercept)
end


####
#### PREDICT FUNCTIONS
####

function glm_predict(coef_, link::GLM.Link, aug_Xmatrix; offset)
    η = aug_Xmatrix * coef_
    isempty(offset) || broadcast!(+, η, η, offset)
    μ = GLM.linkinv.(link, η)
    return μ
end

glm_link(model) = model.link
glm_link(::LinearRegressor) = GLM.IdentityLink()

# more efficient than MLJBase fallback
function MMI.predict_mean(model::GLM_MODELS, fr, Xnew)
    # Note `fr` here is a tuple of `(fitresult, report, cache)`
    fitresult = glm_fitresult(model, fr) # ::FitResult
    coef_ = coef(fitresult)
    link = glm_link(model)
    # `predict_features` here are the features of `Xnew` excluding offsetcol
    Xmatrix, offset, predict_features = prepare_inputs(model, Xnew, false)
    size(Xmatrix, 2) == length(coef_) || throw(
        DimensionMismatch("The number of features in training and test tables must match")
    )
    if model.check_features
        fit_features = params(fitresult).feature_names
        # `fit_features` here are all predictors(excluding offsetcol) 
        # used in fitting the model.
        features_match = fit_features == predict_features
        features_match || throw(
            ArgumentError(
                "The features seen during fitting doesn't "*
                "correspond with those seen during prediction."
            )
        )
    end

    return glm_predict(coef_, link, Xmatrix; offset=offset)
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
    Type{<:LinearRegressor},
    Type{<:LinearBinaryClassifier},
    Type{<:LinearCountRegressor}
}

metadata_pkg.(
    (LinearRegressor, LinearBinaryClassifier, LinearCountRegressor),
    name       = "GLM",
    uuid       = "38e38edf-8417-5370-95a0-9cbb8c7f171a",
    url        = "https://github.com/JuliaStats/GLM.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false
)

metadata_model(
    LinearRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    supports_weights = true,
    descr   = LR_DESCR,
    path    = "$PKG.LinearRegressor"
)

metadata_model(
    LinearBinaryClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite{2}},
    supports_weights = true,
    descr   = LBC_DESCR,
    path    = "$PKG.LinearBinaryClassifier"
)

metadata_model(
    LinearCountRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Count},
    supports_weights = true,
    descr   = LCR_DESCR,
    path    = "$PKG.LinearCountRegressor"
)

end # module
