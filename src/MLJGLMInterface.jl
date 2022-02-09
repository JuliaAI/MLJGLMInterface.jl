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
import Distributions
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


@mlj_model mutable struct LinearRegressor <: MMI.Probabilistic
    fit_intercept::Bool = true
    dropcollinear::Bool = false
    offsetcol::Union{Symbol, Nothing} = nothing
end

@mlj_model mutable struct LinearBinaryClassifier <: MMI.Probabilistic
    fit_intercept::Bool = true
    link::GLM.Link01 = GLM.LogitLink()
    offsetcol::Union{Symbol, Nothing} = nothing
end

@mlj_model mutable struct LinearCountRegressor <: MMI.Probabilistic
    fit_intercept::Bool = true
    distribution::Distributions.Distribution = Distributions.Poisson()
    link::GLM.Link = GLM.LogLink()
    offsetcol::Union{Symbol, Nothing} = nothing
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
    b && return hcat(X, ones(eltype(X), size(X, 1), 1))
    return X
end

_to_vector(v::Vector) = v
_to_vector(v) = collect(v)

"""
    split_X_offset(X, offsetcol::Nothing)

When no offset is specied, return X and an empty vector.
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

"""
    prepare_inputs(model, X; handle_intercept=false)

Handle `model.offsetcol` and `model.fit_intercept` if `handle_intercept=true`.
`handle_intercept` is disabled for fitting since the StatsModels.@formula handles the intercept.
"""
function prepare_inputs(model, X; handle_intercept=false)
    Xminoffset, offset = split_X_offset(X, model.offsetcol)
    Xmatrix = MMI.matrix(Xminoffset)
    if handle_intercept
        Xmatrix = augment_X(Xmatrix, model.fit_intercept)
    end
    return Xmatrix, offset
end

"""
    glm_report(glm_model, features)

Report based on the fitted `LinearModel/GeneralizedLinearModel`, `glm_model`.
"""
function glm_report(glm_model, features)
    deviance = GLM.deviance(glm_model)
    dof_residual = GLM.dof_residual(glm_model)
    stderror = GLM.stderror(glm_model)
    vcov = GLM.vcov(glm_model)
    coef_table = GLM.coeftable(glm_model)
    # Update the variable names in the `coef_table` with the actual variable
    # names seen during fit.
    if length(coef_table.rownms) == length(features)
        # This means `fit_intercept` is false
        coef_table.rownms = string.(features)
    else      
        coef_table.rownms = [
            (string(features[i]) for i in eachindex(features))...; "(Intercept)"
        ]
    end
    return (; deviance, dof_residual, stderror, vcov, coef_table)
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
    header = collect(features)
    data = Tables.table([Xmatrix y]; header=[header; :y])
    return data
end

_to_array(v::AbstractArray) = v
_to_array(v) = collect(v)

"""
    glm_features(model, X)

Returns an iterable features object, to be used in the construction of 
glm formula and glm data header.
"""
function glm_features(model, X)
    if Tables.columnaccess(X)
        table_features = _to_array(keys(Tables.columns(X)))
    else
        first_row = iterate(Tables.rows(X), 1)[1]
        table_features = first_row === nothing ? Symbol[] : _to_array(keys(first_row))
    end
    features = filter!(x -> x != model.offsetcol, table_features)
    return features
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
    function FitResult(
        coefs::AbstractVector,
        dispersion,
        params
    )
        return new{
            typeof(coefs),
            typeof(dispersion),
            typeof(params)
        }(coefs, dispersion, params)
    end
end

coefs(fr::FitResult) = fr.coefs
dispersion(fr::FitResult) = fr.dispersion
params(fr::FitResult) = fr.params

function MMI.fit(model::LinearRegressor, verbosity::Int, X, y)
    # apply the model
    Xmatrix, offset = prepare_inputs(model, X)
    features = glm_features(model, X)
    y_ = isempty(offset) ? y : y .- offset
    data = glm_data(model, Xmatrix, y_, features)
    form = glm_formula(model, features)
    fitted_lm = GLM.lm(form, data; model.dropcollinear).model
    fitresult = FitResult(
        GLM.coef(fitted_lm), GLM.dispersion(fitted_lm), (feature_names = features,)
    )

    # form the report
    report = glm_report(fitted_lm, features)
    cache = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearCountRegressor, verbosity::Int, X, y)
    # apply the model
    Xmatrix, offset = prepare_inputs(model, X)
    features = glm_features(model, X)
    data = glm_data(model, Xmatrix, y, features)
    form = glm_formula(model, features)
    fitted_glm = GLM.glm(form, data, model.distribution, model.link; offset).model
    fitresult = FitResult(
        GLM.coef(fitted_glm), GLM.dispersion(fitted_glm), (feature_names = features,)
    )
    # form the report
    report = glm_report(fitted_glm, features)
    cache = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearBinaryClassifier, verbosity::Int, X, y)
    # apply the model
    decode = y[1]
    y_plain = MMI.int(y) .- 1 # 0, 1 of type Int
    Xmatrix, offset = prepare_inputs(model, X)
    features = glm_features(model, X)
    data = glm_data(model, Xmatrix, y_plain, features)
    form = glm_formula(model, features)
    fitted_glm = GLM.glm(form, data, Distributions.Bernoulli(), model.link; offset).model
    fitresult = FitResult(
        GLM.coef(fitted_glm), GLM.dispersion(fitted_glm), (feature_names = features,)
    )
    # form the report
    report = glm_report(fitted_glm, features)
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
    feature_names = copy(params(result).feature_names)
    if model.fit_intercept
        intercept = coef[end]
        coef_ = coef[1:end-1]
    else
        intercept = zero(eltype(coef))
        coef_ = copy(coef)
    end
    return (; feature_names=feature_names, coef=coef_, intercept=intercept)
end


####
#### PREDICT FUNCTIONS
####

glm_link(model) = model.link
glm_link(::LinearRegressor) = GLM.IdentityLink()

# more efficient than MLJBase fallback
function MMI.predict_mean(model::GLM_MODELS, fitresult, Xnew)
    Xmatrix, offset = prepare_inputs(model, Xnew; handle_intercept=true)
    result = glm_fitresult(model, fitresult) # ::FitResult
    coef = coefs(result)
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
    weights = false,
    descr = LR_DESCR,
    path = "$PKG.LinearRegressor"
)

metadata_model(
    LinearBinaryClassifier,
    input = Table(Continuous),
    target = AbstractVector{<:Finite{2}},
    weights = false,
    descr = LBC_DESCR,
    path = "$PKG.LinearBinaryClassifier"
)

metadata_model(
    LinearCountRegressor,
    input = Table(Continuous),
    target = AbstractVector{Count},
    weights = false,
    descr = LCR_DESCR,
    path = "$PKG.LinearCountRegressor"
)

end # module
