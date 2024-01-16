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
using StatsModels: ConstantTerm, Term, FormulaTerm, term
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

const VALID_KEYS = [
    :deviance,
    :dof_residual,
    :stderror,
    :vcov,
    :coef_table,
    :raw_glm_model,
]
const VALID_KEYS_LIST = join(map(k-> ":$k", VALID_KEYS), ", ", " and ")
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
    augment_X(X::AbstractMatrix, b::Bool)

Augment the matrix `X` with a column of ones if the intercept should
be fitted (`b==true`) and return `X` otherwise.
"""
function augment_X(X::AbstractMatrix, b::Bool)
    if b
        return hcat(X, ones(float(Int), size(X, 1), 1))
    else
        return X
    end
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
    if :raw_glm_model in reportkeys
        report_dict[:raw_glm_model] = glm_model
    end

    return NamedTuple{Tuple(keys(report_dict))}(values(report_dict))
end

function _ensure_intercept_at_end(form::FormulaTerm)
    fixed_rhs = if first(form.rhs) isa ConstantTerm
        (form.rhs[2:end]..., form.rhs[1])
    else
        form.rhs
    end
    return FormulaTerm(form.lhs, fixed_rhs)
end

"""
    glm_formula(model, features::AbstractVector{Symbol}) -> FormulaTerm

Return formula which is ready to be passed to `fit(form, data, ...)`.
"""
function glm_formula(model, features::AbstractVector{Symbol})::FormulaTerm
    # By default, using a JuliaStats formula will add an intercept.
    # Adding a zero term explicitly disables the intercept.
    # See the StatsModels.jl tests for more information.
    intercept_term = model.fit_intercept ? 1 : 0
    form = FormulaTerm(Term(:y), sum(term.(features)) + term(intercept_term))
    fixed_form = _ensure_intercept_at_end(form)
    return fixed_form
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

coefs(fr::FitResult) = fr.coefs

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
    path = "$PKG.LinearRegressor"
)

metadata_model(
    LinearBinaryClassifier,
    input = Table(Continuous),
    target = AbstractVector{<:Finite{2}},
    supports_weights = true,
    path = "$PKG.LinearBinaryClassifier"
)

metadata_model(
    LinearCountRegressor,
    input = Table(Continuous),
    target = AbstractVector{Count},
    supports_weights = true,
    path = "$PKG.LinearCountRegressor"
)

"""
$(MMI.doc_header(LinearRegressor))

`LinearRegressor` assumes the target is a continuous variable
whose conditional distribution is normal with constant variance, and whose
expected value is a linear combination of the features (identity link function).
Options exist to specify an intercept or offset feature.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with one of:

    mach = machine(model, X, y)
    mach = machine(model, X, y, w)

Here

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

- `w`: is a vector of `Real` per-observation weights

# Hyper-parameters

- `fit_intercept=true`: Whether to calculate the intercept for this model.
   If set to false, no intercept will be calculated (e.g. the data is expected
   to be centered)

- `dropcollinear=false`: Whether to drop features in the training data
  to ensure linear independence.  If true , only the first of each set
  of linearly-dependent features is used. The coefficient for
  redundant linearly dependent features is `0.0` and all associated
  statistics are set to `NaN`.

- `offsetcol=nothing`: Name of the column to be used as an offset, if any.
   An offset is a variable which is known to have a coefficient of 1.

- `report_keys::Union{Symbol, Nothing}=DEFAULT_KEYS`: vector of keys to be used in
  the report. Possible keys are: $VALID_KEYS_LIST.

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
   features `Xnew` having the same Scitype as `X` above. Predictions are
   probabilistic.

- `predict_mean(mach, Xnew)`: instead return the mean of
   each prediction above

- `predict_median(mach, Xnew)`: instead return the median of
   each prediction above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features`: The names of the features encountered during model fitting.

- `coef`: The linear coefficients determined by the model.

- `intercept`: The intercept determined by the model.

# Report

When all keys are enabled in `report_keys`, the following fields are available in
`report(mach)`:

- `deviance`: Measure of deviance of fitted model with respect to
  a perfectly fitted model. For a linear model, this is the weighted
  residual sum of squares

- `dof_residual`: The degrees of freedom for residuals, when meaningful.

- `stderror`: The standard errors of the coefficients.

- `vcov`: The estimated variance-covariance matrix of the coefficient estimates.

- `coef_table`: Table which displays coefficients and summarizes their significance
  and confidence intervals.

- `raw_glm_model`: The raw fitted model returned by `GLM.lm`. Note this points to training
  data.

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

`LinearBinaryClassifier` is a [generalized linear
model](https://en.wikipedia.org/wiki/Generalized_linear_model#Variance_function),
specialised to the case of a binary target variable, with a user-specified link function.
Options exist to specify an intercept or offset feature.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with one of:

    mach = machine(model, X, y)
    mach = machine(model, X, y, w)

Here

- `X`: is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element scitype is
  `<:OrderedFactor(2)` or `<:Multiclass(2)`; check the scitype with `schema(y)`

- `w`: is a vector of `Real` per-observation weights

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `fit_intercept=true`: Whether to calculate the intercept for this model.  If set to false,
   no intercept will be calculated (e.g. the data is expected to be centered)

- `link=GLM.LogitLink`: The function which links the linear prediction function to the
   probability of a particular outcome or class. This must have type `GLM.Link01`. Options
   include `GLM.LogitLink()`, `GLM.ProbitLink()`, `CloglogLink(), `CauchitLink()`.

- `offsetcol=nothing`: Name of the column to be used as an offset, if any.  An offset is a
   variable which is known to have a coefficient of 1.

- `maxiter::Integer=30`: The maximum number of iterations allowed to achieve convergence.

- `atol::Real=1e-6`: Absolute threshold for convergence. Convergence is achieved when the
   relative change in deviance is less than `max(rtol*dev, atol). This term exists to avoid
   failure when deviance is unchanged except for rounding errors.

- `rtol::Real=1e-6`: Relative threshold for convergence. Convergence is achieved when the
   relative change in deviance is less than `max(rtol*dev, atol). This term exists to avoid
   failure when deviance is unchanged except for rounding errors.

- `minstepfac::Real=0.001`: Minimum step fraction. Must be between 0 and 1. Lower bound for
  the factor used to update the linear fit.

- `report_keys::Union{Symbol, Nothing}=DEFAULT_KEYS`: keys to be used in the
  report. Possible keys are: $VALID_KEYS_LIST.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew` having the
  same scitype as `X` above. Predictions are probabilistic.

- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions returned
   above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features`: The names of the features used during model fitting.

- `coef`: The linear coefficients determined by the model.

- `intercept`: The intercept determined by the model.

# Report

The fields of `report(mach)` are:

- `deviance`: Measure of deviance of fitted model with respect to a perfectly fitted
  model. For a linear model, this is the weighted residual sum of squares

- `dof_residual`: The degrees of freedom for residuals, when meaningful.

- `stderror`: The standard errors of the coefficients.

- `vcov`: The estimated variance-covariance matrix of the coefficient estimates.

- `coef_table`: Table which displays coefficients and summarizes their significance and
  confidence intervals.

- `raw_glm_model`: The raw fitted model returned by `GLM.glm`. Note this points to training
  data.

# Examples

```
using MLJ
import GLM # namespace must be available

LinearBinaryClassifier = @load LinearBinaryClassifier pkg=GLM
clf = LinearBinaryClassifier(fit_intercept=false, link=GLM.ProbitLink())

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

`LinearCountRegressor` is a [generalized linear
model](https://en.wikipedia.org/wiki/Generalized_linear_model#Variance_function),
specialised to the case of a `Count` target variable (non-negative, unbounded integer) with
user-specified link function. Options exist to specify an intercept or offset feature.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with one of:

    mach = machine(model, X, y)
    mach = machine(model, X, y, w)

Here

- `X`: is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element scitype is `Count`;
  check the scitype with `schema(y)`

- `w`: is a vector of `Real` per-observation weights

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `fit_intercept=true`: Whether to calculate the intercept for this model. If set to false,
   no intercept will be calculated (e.g. the data is expected to be centered)

- `distribution=Distributions.Poisson()`: The distribution which the residuals/errors of the
   model should fit.

- `link=GLM.LogLink()`: The function which links the linear prediction function to the
   probability of a particular outcome or class. This should be one of the following:
   `GLM.IdentityLink()`, `GLM.InverseLink()`, `GLM.InverseSquareLink()`, `GLM.LogLink()`,
   `GLM.SqrtLink()`.

- `offsetcol=nothing`: Name of the column to be used as an offset, if any.  An offset is a
   variable which is known to have a coefficient of 1.

- `maxiter::Integer=30`: The maximum number of iterations allowed to achieve convergence.

- `atol::Real=1e-6`: Absolute threshold for convergence. Convergence is achieved when the
   relative change in deviance is less than `max(rtol*dev, atol). This term exists to avoid
   failure when deviance is unchanged except for rounding errors.

- `rtol::Real=1e-6`: Relative threshold for convergence. Convergence is achieved when the
   relative change in deviance is less than `max(rtol*dev, atol). This term exists to avoid
   failure when deviance is unchanged except for rounding errors.

- `minstepfac::Real=0.001`: Minimum step fraction. Must be between 0 and 1. Lower bound for
  the factor used to update the linear fit.

- `report_keys::Union{Symbol, Nothing}=DEFAULT_KEYS`: keys to be used in the
  report. Possible keys are: $VALID_KEYS_LIST.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features `Xnew` having
   the same Scitype as `X` above. Predictions are probabilistic.

- `predict_mean(mach, Xnew)`: instead return the mean of each prediction above

- `predict_median(mach, Xnew)`: instead return the median of each prediction above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features`: The names of the features encountered during model fitting.

- `coef`: The linear coefficients determined by the model.

- `intercept`: The intercept determined by the model.

# Report

The fields of `report(mach)` are:

- `deviance`: Measure of deviance of fitted model with respect to a perfectly fitted
  model. For a linear model, this is the weighted residual sum of squares

- `dof_residual`: The degrees of freedom for residuals, when meaningful.

- `stderror`: The standard errors of the coefficients.

- `vcov`: The estimated variance-covariance matrix of the coefficient estimates.

- `coef_table`: Table which displays coefficients and summarizes their significance and
  confidence intervals.

- `raw_glm_model`: The raw fitted model returned by `GLM.glm`. Note this points to training
  data.


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
