using Test

using MLJBase
using LinearAlgebra
using Statistics
using MLJGLMInterface
import GLM

import Distributions
import StableRNGs
using Tables


expit(X) = 1 ./ (1 .+ exp.(-X))

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
@test hyperparameters(model) == (:fit_intercept, :allowrankdeficient, :offsetcol)
@test hyperparameter_types(model) == ("Bool", "Bool", "Union{Nothing, Symbol}")

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


modeltypes = [LinearRegressor, LinearBinaryClassifier, LinearCountRegressor]
@testset "Test prepare_inputs" begin
    @testset "intercept/offsetcol" for mt in modeltypes
            X = (x1=[1,2,3], x2=[4,5,6])
            m = mt(fit_intercept=true, offsetcol=:x2)
            Xmatrix, offset = MLJGLMInterface.prepare_inputs(m, X)

            @test offset == [4, 5, 6]
            @test Xmatrix== [1 1;
                             2 1;
                             3 1]
    end

    @testset "no intercept/no offsetcol" for mt in modeltypes
        X = (x1=[1,2,3], x2=[4,5,6])
        m = mt(fit_intercept=false)
        Xmatrix, offset = MLJGLMInterface.prepare_inputs(m, X)

        @test offset == []
        @test Xmatrix == [1 4;
                          2 5;
                          3 6]
    end

end


@testset "Test offsetting models" begin
    @testset "Test split_X_offset" begin
        X = (x1=[1,2,3], x2=[4,5,6])
        @test MLJGLMInterface.split_X_offset(X, nothing) == (X, Float64[])
        @test MLJGLMInterface.split_X_offset(X, :x1) == ((x2=[4,5,6],), [1,2,3])

        X = MLJBase.table(rand(rng, N, 3))
        Xnew, offset = MLJGLMInterface.split_X_offset(X, :x2)
        @test offset isa Vector
        @test length(Xnew) == 2
    end

    # In the following:
    # The second column is taken as an offset by the model
    # This is equivalent to assuming the coef is 1 and known 
    
    @testset "Test Logistic regression with offset" begin
        N = 1000
        rng = StableRNGs.StableRNG(0)
        X = MLJBase.table(rand(rng, N, 3))
        y = rand(rng, Distributions.Uniform(0,1), N) .< expit(2*X.x1 + X.x2 - X.x3)
        y = categorical(y)

        lr = LinearBinaryClassifier(fit_intercept=false, offsetcol=:x2)
        fitresult, _, report = fit(lr, 1, X, y)
        fp = fitted_params(lr, fitresult)

        @test fp.coef ≈ [2, -1] atol=0.03
    end
    @testset "Test Linear regression with offset" begin
        N = 1000
        rng = StableRNGs.StableRNG(0)
        X = MLJBase.table(rand(rng, N, 3))
        y = 2*X.x1 + X.x2 - X.x3 + rand(rng, Distributions.Normal(0,1), N) 

        lr = LinearRegressor(fit_intercept=false, offsetcol=:x2)
        fitresult, _, report = fit(lr, 1, X, y)
        fp = fitted_params(lr, fitresult)

        @test fp.coef ≈ [2, -1] atol=0.07
    end
    @testset "Test Count regression with offset" begin
        N = 1000
        rng = StableRNGs.StableRNG(0)
        X = MLJBase.table(rand(rng, N, 3))
        y = map(exp.(2*X.x1 + X.x2 - X.x3)) do mu
            rand(rng, Distributions.Poisson(mu))
        end

        lcr = LinearCountRegressor(fit_intercept=false, offsetcol=:x2)
        fitresult, _, _ = fit(lcr, 1, X, y)
        fp = fitted_params(lcr, fitresult)

        @test fp.coef ≈ [2, -1] atol=0.04
    end
end