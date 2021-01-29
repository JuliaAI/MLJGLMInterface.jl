using Test

using MLJBase
using LinearAlgebra
using Statistics
using MLJGLMInterface
import GLM

import Distributions
import StableRNGs
using Tables

###
### OLSREGRESSOR
###

X, y = @load_boston

train, test = partition(eachindex(y), 0.7)

atom_ols = LinearRegressor()

Xtrain = selectrows(X, train)
ytrain = selectrows(y, train)
Xtest  = selectrows(X, test)

fitresult, _, report = fit(atom_ols, 1, Xtrain, ytrain)
θ = MLJBase.fitted_params(atom_ols, fitresult)

p = predict_mean(atom_ols, fitresult, Xtest)

# hand made regression to compare

Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
Xa1 = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]
p2 = Xa1[test, :] * coefs

@test p ≈ p2

model = atom_ols
@test name(model) == "LinearRegressor"
@test package_name(model) == "GLM"
@test is_pure_julia(model)
@test is_supervised(model)
@test package_license(model) == "MIT"
@test prediction_type(model) == :probabilistic
@test hyperparameters(model) == (:fit_intercept, :allowrankdeficient)
@test hyperparameter_types(model) == ("Bool", "Bool")

p_distr = predict(atom_ols, fitresult, selectrows(X, test))

@test p_distr[1] == Distributions.Normal(p[1], GLM.dispersion(fitresult))

###
### Logistic regression
###

rng = StableRNGs.StableRNG(0)

N = 100
X = MLJBase.table(rand(rng, N, 4));
ycont = 2*X.x1 - X.x3 + 0.6*rand(rng, N)
y = (ycont .> mean(ycont)) |> categorical;

lr = LinearBinaryClassifier()
fitresult, _, report = fit(lr, 1, X, y)

yhat = predict(lr, fitresult, X)
@test mean(cross_entropy(yhat, y)) < 0.25

pr = LinearBinaryClassifier(link=GLM.ProbitLink())
fitresult, _, report = fit(pr, 1, X, y)
yhat = predict(lr, fitresult, X)
@test mean(cross_entropy(yhat, y)) < 0.25

fitted_params(pr, fitresult)


###
### Count regression
###

rng = StableRNGs.StableRNG(123)

X = randn(rng, 500, 5)
θ = randn(rng, 5)
y = map(exp.(X*θ)) do mu
    rand(rng, Distributions.Poisson(mu))
end

XTable = MLJBase.table(X)

lcr = LinearCountRegressor(fit_intercept=false)
fitresult, _, _ = fit(lcr, 1, XTable, y)

θ̂ = fitted_params(lcr, fitresult).coef

@test norm(θ̂ .- θ)/norm(θ) ≤ 0.03
