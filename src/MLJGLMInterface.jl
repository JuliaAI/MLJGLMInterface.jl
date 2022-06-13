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
    allowrankdeficient::Bool = false
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
    return newX, Vector(offset)
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
    glm_report(fitresult)

Report based on the `fitresult` of a GLM model.
"""
function glm_report(fitresult)
    deviance = GLM.deviance(fitresult)
    dof_residual = GLM.dof_residual(fitresult)
    stderror = GLM.stderror(fitresult)
    vcov = GLM.vcov(fitresult)
    return (; deviance=deviance, dof_residual=dof_residual, stderror=stderror, vcov=vcov)
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

"""
    glm_features(model, X)

Returns an iterable features object, to be used in the construction of
glm formula and glm data header.
"""
function glm_features(model, X)
    if Tables.columnaccess(X)
        table_features = keys(Tables.columns(X))
    else
        first_row = iterate(Tables.rows(X), 1)[1]
        table_features = first_row === nothing ? Symbol[] : keys(first_row)
    end
    features = filter(x -> x != model.offsetcol, table_features)
    return features
end

####
#### FIT FUNCTIONS
####


function MMI.fit(model::LinearRegressor, verbosity::Int, X, y)
    # apply the model
    Xmatrix, offset = prepare_inputs(model, X)
    features = glm_features(model, X)
    data = glm_data(model, Xmatrix, y, features)
    form = glm_formula(model, features)
    fitresult = GLM.glm(
        form, data, Distributions.Normal(), GLM.IdentityLink();
        offset=offset
    )
    # form the report
    report = glm_report(fitresult)
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
    fitresult = GLM.glm(
        form, data, model.distribution, model.link;
        offset=offset
    )
    # form the report
    report = glm_report(fitresult)
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
    fitresult = GLM.glm(
        form, data, Distributions.Bernoulli(), model.link;
        offset=offset
    )
    # form the report
    report = glm_report(fitresult)
    cache = nothing
    # return
    return (fitresult, decode), cache, report
end

glm_fitresult(::LinearRegressor, fitresult) = fitresult
glm_fitresult(::LinearCountRegressor, fitresult) = fitresult
glm_fitresult(::LinearBinaryClassifier, fitresult) = fitresult[1]

function MMI.fitted_params(model::GLM_MODELS, fitresult)
    result = glm_fitresult(model, fitresult)
    coef = GLM.coef(result)
    features = filter(name -> name != "(Intercept)", GLM.coefnames(result))
    intercept = model.fit_intercept ? coef[end] : nothing
    return (; features=features, coef=coef, intercept=intercept)
end


####
#### PREDICT FUNCTIONS
####

# more efficient than MLJBase fallback
function MMI.predict_mean(model::Union{LinearRegressor,<:LinearCountRegressor}, fitresult, Xnew)
    Xmatrix, offset = prepare_inputs(model, Xnew; handle_intercept=true)
    return GLM.predict(fitresult, Xmatrix; offset=offset)
end

function MMI.predict_mean(model::LinearBinaryClassifier, (fitresult, _), Xnew)
    Xmatrix, offset = prepare_inputs(model, Xnew; handle_intercept=true)
    return GLM.predict(fitresult.model, Xmatrix; offset=offset)
end

function MMI.predict(model::LinearRegressor, fitresult, Xnew)
    μ = MMI.predict_mean(model, fitresult, Xnew)
    σ̂ = GLM.dispersion(fitresult.model)
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

"""
$(MMI.doc_header(LinearRegressor))

`LinearRegressor` assumes the target is a continuous variable
whose conditional distribution is normal with constant variance, and whose
expected value is a linear combination of the features (identity link function).
Options exist to specify an intercept or offset feature.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

# Hyper-parameters

- `fit_intercept=true`: Whether to calculate the intercept for this model.
   If set to false, no intercept will be calculated (e.g. the data is expected
   to be centered)
- `allowrankdeficient=false`: Whether to allow rank deficient matrices (e.g.
   more columns than rows)
- `offsetcol=nothing`: Name of the column to be used as an offset, if any.
   An offset is a variable which is known to have a coefficient of 1.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
   features `Xnew` having the same Scitype as `X` above. Predictions are
   probabilistic.
- `predict_mean(mach, Xnew)`: instead return the mean of
   each prediction above

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features`: The names of the features used during model fitting.
- `coef`: The linear coefficients determined by the model.
- `intercept`: The intercept determined by the model.

# Report

The fields of `report(mach)` are:

- `deviance`: Measure of deviance of fitted model with respect to
  a perfectly fitted model. For a linear model, this is the weighted
  residual sum of squares
- `dof_residual`: The degrees of freedom for residuals, when meaningful.
- `stderror`: The standard errors of the coefficients.
- `vcov`: The estimated variance-covariance matrix of the coefficient estimates.
- `coef_table`: Table which displays coefficients and summarizes their significance
  and confidence intervals.

# Examples

```
using MLJ
LinearRegressor = @load LinearRegressor pkg=GLM
glm = LinearRegressor()

X, y = make_regression(100, 2) # synthetic data
mach = machine(glm, X, y) |> fit!

Xnew, _ = make_regression(3, 2)
yhat = predict(mach, Xnew) # new predictions
yhat_point = predict_mean(mach, Xnew) # new predictions

fitted_params(mach).features
fitted_params(mach).coef # x1, x2, intercept
fitted_params(mach).intercept

report(mach)
```

See also
[`LinearCountRegressor`](@ref), [`LinearBinaryClassifier`](@ref)
"""
LinearRegressor

"""
$(MMI.doc_header(LinearBinaryClassifier))

`LinearBinaryClassifier` is a [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model#Variance_function), specialised
to the case of a binary target variable, with a user-specified link function. 
Options exist to specify an intercept or offset feature.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor(2)` or `<:Multiclass(2)`; check the scitype
  with `schema(y)`

# Hyper-parameters

- `fit_intercept=true`: Whether to calculate the intercept for this model.
   If set to false, no intercept will be calculated (e.g. the data is expected
   to be centered)
- `link=GLM.LogitLink`: The function which links the linear prediction function
   to the probability of a particular outcome or class.T his must have type `GLM.Link01`. Options include
   `GLM.LogitLink()`, `GLM.ProbitLink()`, `CloglogLink(), `CauchitLink()`.
- `offsetcol=nothing`: Name of the column to be used as an offset, if any.
   An offset is a variable which is known to have a coefficient of 1.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic.
- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions
   returned above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features`: The names of the features used during model fitting.
- `coef`: The linear coefficients determined by the model.
- `intercept`: The intercept determined by the model.

# Report

The fields of `report(mach)` are:

- `deviance`: Measure of deviance of fitted model with respect to
  a perfectly fitted model. For a linear model, this is the weighted
  residual sum of squares
- `dof_residual`: The degrees of freedom for residuals, when meaningful.
- `stderror`: The standard errors of the coefficients.
- `vcov`: The estimated variance-covariance matrix of the coefficient estimates.
- `coef_table`: Table which displays coefficients and summarizes their significance
  and confidence intervals.

# Examples

```
using MLJ
import GLM # namespace must be available

CLF = @load LinearBinaryClassifier pkg=GLM
clf = CLF(fit_intercept=false, link=GLM.ProbitLink())

X, y = @load_crabs

mach = machine(clf, X, y) |> fit!

Xnew = (;FL = [8.1, 24.8, 7.2],
        RW = [5.1, 25.7, 6.4],
        CL = [15.9, 46.7, 14.3],
        CW = [18.7, 59.7, 12.2],
        BD = [6.2, 23.6, 8.4],)

yhat = predict(mach, Xnew) # probabilistic predictions
pdf(yhat, levels(y)) # probability matrix
p_B = pdf.(yhat, "B")
class_labels = predict_mode(mach, Xnew)


fitted_params(mach).features
fitted_params(mach).coef
fitted_params(mach).intercept

report(mach)
```

See also
[`LinearRegressor`](@ref), [`LinearCountRegressor`](@ref)
"""
LinearBinaryClassifier

"""
$(MMI.doc_header(LinearCountRegressor))

`LinearCountRegressor` is a [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model#Variance_function), specialised
to the case of a `Count` target variable (non-negative, unbounded integer) with user-specified
link function. Options exist to
specify an intercept or offset feature.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Count`; check the scitype with `schema(y)`

# Hyper-parameters

- `fit_intercept=true`: Whether to calculate the intercept for this model.
   If set to false, no intercept will be calculated (e.g. the data is expected
   to be centered)
- `distribution=Distributions.Poisson()`: The distribution which the residuals/errors 
   of the model should fit.
- `link=GLM.LogLink()`: The function which links the linear prediction function
   to the probability of a particular outcome or class. This should be one of the following:
   `GLM.IdentityLink()`, `GLM.InverseLink()`, `GLM.InverseSquareLink()`,
   `GLM.LogLink()`, `GLM.SqrtLink()`.
- `offsetcol=nothing`: Name of the column to be used as an offset, if any.
   An offset is a variable which is known to have a coefficient of 1.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
   features `Xnew` having the same Scitype as `X` above. Predictions are
   probabilistic.
- `predict_mean(mach, Xnew)`: instead return the mean of
   each prediction above
- `predict_median(mach, Xnew)`: instead return the median values.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features`: The names of the features used during model fitting.
- `coef`: The linear coefficients determined by the model.
- `intercept`: The intercept determined by the model.

# Report

The fields of `report(mach)` are:

- `deviance`: Measure of deviance of fitted model with respect to
  a perfectly fitted model. For a linear model, this is the weighted
  residual sum of squares
- `dof_residual`: The degrees of freedom for residuals, when meaningful.
- `stderror`: The standard errors of the coefficients.
- `vcov`: The estimated variance-covariance matrix of the coefficient estimates.
- `coef_table`: Table which displays coefficients and summarizes their significance
  and confidence intervals.


# Examples

```
using MLJ
import MLJ.Distributions.Poisson

# Generate some data whose target y looks Poisson when conditioned on
# X:
N = 10_000
w = [1.0, -2.0, 3.0]
mu(x) = exp(w'x) # mean for a log link function
Xmat = rand(N, 3)
X = MLJ.table(Xmat)
y = map(1:N) do i
    x = Xmat[i, :]
    rand(Poisson(mu(x)))
end;

CountRegressor = @load LinearCountRegressor pkg=GLM
model = CountRegressor(fit_intercept=false)
mach = machine(model, X, y)
fit!(mach)

Xnew = MLJ.table(rand(3, 3))
yhat = predict(mach, Xnew)
yhat_point = predict_mean(mach, Xnew)

# get coefficients approximating `w`:
julia> fitted_params(mach).coef
3-element Vector{Float64}:
  0.9969008753103842
 -2.0255901752504775
  3.014407534033522

report(mach)
```

See also
[`LinearRegressor`](@ref), [`LinearBinaryClassifier`](@ref)
"""
LinearCountRegressor

end # module
