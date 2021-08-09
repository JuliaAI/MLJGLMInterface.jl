module MLJGLMInterface

# -------------------------------------------------------------------
# TODO
# - return feature names in the report
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
using Parameters

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

"""
When no offset is specied, return X and an empty vector
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
    return newX, Vector(offset)
end


function prepare_inputs(model, X)
    Xminoffset, offset = split_X_offset(X, model.offsetcol)
    Xmatrix   = augment_X(MMI.matrix(Xminoffset), model.fit_intercept)
    return Xmatrix, offset
end

"""
glm_report(fitresult)

Report based on the `fitresult` of a GLM model.
"""
glm_report(fitresult) = ( deviance     = GLM.deviance(fitresult),
                          dof_residual = GLM.dof_residual(fitresult),
                          stderror     = GLM.stderror(fitresult),
                          vcov         = GLM.vcov(fitresult) )

####
#### REGRESSION TYPES
####

# LinearRegressor        --> Probabilistic w Continuous Target
# LinearCountRegressor   --> Probabilistic w Count Target
# LinearBinaryClassifier --> Probabilistic w Binary target // logit,cauchit,..
# MulticlassClassifier   --> Probabilistic w Multiclass target


@with_kw_noshow mutable struct LinearRegressor <: MMI.Probabilistic
    fit_intercept::Bool                 = true
    allowrankdeficient::Bool            = false
    offsetcol::Union{Symbol, Nothing}   = nothing
end

@with_kw_noshow mutable struct LinearBinaryClassifier{L<:GLM.Link01} <: MMI.Probabilistic
    fit_intercept::Bool                 = true
    link::L                             = GLM.LogitLink()
    offsetcol::Union{Symbol, Nothing}   = nothing
end

@with_kw_noshow mutable struct LinearCountRegressor{D<:Distributions.Distribution,L<:GLM.Link} <: MMI.Probabilistic
    fit_intercept::Bool                 = true
    distribution::D                     = Distributions.Poisson()
    link::L                             = GLM.LogLink()
    offsetcol::Union{Symbol, Nothing}   = nothing
end

# Short names for convenience here

const GLM_MODELS = Union{<:LinearRegressor, <:LinearBinaryClassifier, <:LinearCountRegressor}

####
#### FIT FUNCTIONS
####

function MMI.fit(model::LinearRegressor, verbosity::Int, X, y)
    # apply the model
    Xmatrix, offset = prepare_inputs(model, X)
    fitresult = GLM.glm(Xmatrix, y, Distributions.Normal(), GLM.IdentityLink(); offset=offset)
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearCountRegressor, verbosity::Int, X, y)
    # apply the model
    Xmatrix, offset = prepare_inputs(model, X)
    fitresult = GLM.glm(Xmatrix, y, model.distribution, model.link; offset=offset)
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearBinaryClassifier, verbosity::Int, X, y)
    # apply the model
    Xmatrix, offset = prepare_inputs(model, X)
    decode    = y[1]
    y_plain   = MMI.int(y) .- 1 # 0, 1 of type Int
    fitresult = GLM.glm(Xmatrix, y_plain, Distributions.Bernoulli(), model.link; offset=offset)
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return (fitresult, decode), cache, report
end

glm_fitresult(::LinearRegressor, fitresult)  = fitresult
glm_fitresult(::LinearCountRegressor, fitresult)  = fitresult
glm_fitresult(::LinearBinaryClassifier, fitresult)  = fitresult[1]

function MMI.fitted_params(model::GLM_MODELS, fitresult)
    coefs = GLM.coef(glm_fitresult(model,fitresult))
    return (coef      = coefs[1:end-Int(model.fit_intercept)],
            intercept = ifelse(model.fit_intercept, coefs[end], nothing))
end


####
#### PREDICT FUNCTIONS
####

# more efficient than MLJBase fallback
function MMI.predict_mean(model::Union{LinearRegressor,<:LinearCountRegressor}, fitresult, Xnew)
    Xmatrix, offset = prepare_inputs(model, Xnew)
    return GLM.predict(fitresult, Xmatrix; offset=offset)
end

function MMI.predict_mean(model::LinearBinaryClassifier, (fitresult, _), Xnew)
    Xmatrix, offset = prepare_inputs(model, Xnew)
    return GLM.predict(fitresult, Xmatrix; offset=offset)
end

function MMI.predict(model::LinearRegressor, fitresult, Xnew)
    μ = MMI.predict_mean(model, fitresult, Xnew)
    σ̂ = GLM.dispersion(fitresult)
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
const GLM_REGS = Union{Type{<:LinearRegressor},
                       Type{<:LinearBinaryClassifier},
                       Type{<:LinearCountRegressor}}

metadata_pkg.((LinearRegressor, LinearBinaryClassifier, LinearCountRegressor),
              name       = "GLM",
              uuid       = "38e38edf-8417-5370-95a0-9cbb8c7f171a",
              url        = "https://github.com/JuliaStats/GLM.jl",
              julia      = true,
              license    = "MIT",
              is_wrapper = false
              )

metadata_model(LinearRegressor,
               input   = Table(Continuous),
               target  = AbstractVector{Continuous},
               weights = false,
               descr   = LR_DESCR,
               path    = "$PKG.LinearRegressor"
               )

metadata_model(LinearBinaryClassifier,
               input   = Table(Continuous),
               target  = AbstractVector{<:Finite{2}},
               weights = false,
               descr   = LBC_DESCR,
               path    = "$PKG.LinearBinaryClassifier"
               )

metadata_model(LinearCountRegressor,
               input   = Table(Continuous),
               target  = AbstractVector{Count},
               weights = false,
               descr   = LCR_DESCR,
               path    = "$PKG.LinearCountRegressor"
               )

end # module
