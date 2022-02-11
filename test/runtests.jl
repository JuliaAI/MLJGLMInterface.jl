using Test

using MLJBase
using LinearAlgebra
using Statistics
using MLJGLMInterface
using GLM: coeftable
import GLM

import Distributions
import StableRNGs
using Tables


expit(X) = 1 ./ (1 .+ exp.(-X))

###
### OLSREGRESSOR
###
@testset "OLSREGRESSOR" begin
    X, y = @load_boston
    w = ones(eltype(y), length(y))

    train, test = partition(eachindex(y), 0.7)

    atom_ols = LinearRegressor()

    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    wtrain = selectrows(w, train)
    Xtest  = selectrows(X, test)

    # Without weights `w`
    fitresult, _, _ = fit(atom_ols, 1, Xtrain, ytrain)
    θ = MLJBase.fitted_params(atom_ols, fitresult)

    p = predict_mean(atom_ols, fitresult, Xtest)

    # With weights `w`
    fitresultw, _, _ = fit(atom_ols, 1, Xtrain, ytrain, wtrain)
    θw = MLJBase.fitted_params(atom_ols, fitresult)

    pw = predict_mean(atom_ols, fitresultw, Xtest)

    # hand made regression to compare
    Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
    Xa1 = hcat(Xa, ones(size(Xa, 1)))
    coefs = Xa1[train, :] \ y[train]
    p2 = Xa1[test, :] * coefs

    @test p ≈ p2
    @test pw ≈ p2

    # test `predict` object
    p_distr = predict(atom_ols, fitresult, selectrows(X, test))
    dispersion =  MLJGLMInterface.dispersion(fitresult)
    @test p_distr[1] == Distributions.Normal(p[1], dispersion)

    # test metadata
    model = atom_ols
    @test name(model) == "LinearRegressor"
    @test package_name(model) == "GLM"
    @test supports_weights(model)
    @test is_pure_julia(model)
    @test is_supervised(model)
    @test package_license(model) == "MIT"
    @test prediction_type(model) == :probabilistic
    @test hyperparameters(model) == (:fit_intercept, :dropcollinear, :offsetcol, :report_keys)
    hyp_types = hyperparameter_types(model)
    @test hyp_types[1] == "Bool"
    @test hyp_types[2] == "Bool"
    @test hyp_types[3] == "Union{Nothing, Symbol}"
    @test hyp_types[4] == "Union{Nothing, AbstractVector{Symbol}}"
    
end

###
### Logistic regression
###
@testset "Logistic regression" begin
    rng = StableRNGs.StableRNG(0)

    N = 100
    X = MLJBase.table(rand(rng, N, 4));
    ycont = 2*X.x1 - X.x3 + 0.6*rand(rng, N)
    y = categorical(ycont .> mean(ycont))
    w = ones(length(y))

    lr = LinearBinaryClassifier()
    pr = LinearBinaryClassifier(link=GLM.ProbitLink())

    # without weights
    fitresult, _, report = fit(lr, 1, X, y)
    yhat = predict(lr, fitresult, X)
    @test mean(cross_entropy(yhat, y)) < 0.25
    fitresult1, _, report1 = fit(pr, 1, X, y)
    yhat1 = predict(pr, fitresult1, X)
    @test mean(cross_entropy(yhat1, y)) < 0.25

    # With weights
    fitresultw, _, reportw = fit(lr, 1, X, y, w)
    yhatw = predict(lr, fitresultw, X)
    @test mean(cross_entropy(yhatw, y)) < 0.25
    @test yhatw ≈ yhat
    fitresultw1, _, reportw1 = fit(pr, 1, X, y, w)
    yhatw1 = predict(pr, fitresultw1, X)
    @test mean(cross_entropy(yhatw1, y)) < 0.25
    @test yhatw1 ≈ yhat1

    fitted_params(pr, fitresult)

    # Test metadata
    model = lr
    @test name(model) == "LinearBinaryClassifier"
    @test package_name(model) == "GLM"
    @test supports_weights(model)
    @test is_pure_julia(model)
    @test is_supervised(model)
    @test package_license(model) == "MIT"
    @test prediction_type(model) == :probabilistic
    @test hyperparameters(model) == (:fit_intercept, :link, :offsetcol, :report_keys)
end

###
### Count regression
###
@testset "Count regression" begin
    rng = StableRNGs.StableRNG(123)

    X = randn(rng, 500, 5)
    θ = randn(rng, 5)
    y = map(exp.(X*θ)) do mu
        rand(rng, Distributions.Poisson(mu))
    end
    w = ones(eltype(y), length(y))

    XTable = MLJBase.table(X)

    lcr = LinearCountRegressor(fit_intercept=false)

    # Without weights 
    fitresult, _, _ = fit(lcr, 1, XTable, y)
    θ̂ = fitted_params(lcr, fitresult).coef
    @test norm(θ̂ .- θ)/norm(θ) ≤ 0.03

    # With weights
    fitresultw, _, _ = fit(lcr, 1, XTable, y, w)
    θ̂w = fitted_params(lcr, fitresultw).coef
    @test norm(θ̂w .- θ)/norm(θ) ≤ 0.03
    @test θ̂w ≈ θ̂ 

    # Test metadata
    model = lcr
    @test name(model) == "LinearCountRegressor"
    @test package_name(model) == "GLM"
    @test supports_weights(model)
    @test is_pure_julia(model)
    @test is_supervised(model)
    @test package_license(model) == "MIT"
    @test prediction_type(model) == :probabilistic
    hyper_params = hyperparameters(model)
    @test hyper_params[1] == :fit_intercept
    @test hyper_params[2] == :distribution
    @test hyper_params[3] == :link
    @test hyper_params[4] == :offsetcol
    @test hyper_params[5] == :report_keys

end

modeltypes = [LinearRegressor, LinearBinaryClassifier, LinearCountRegressor]
@testset "Test prepare_inputs" begin
    @testset "intercept/offsetcol" for mt in modeltypes
            X = (x1=[1,2,3], x2=[4,5,6])
            m = mt(fit_intercept=true, offsetcol=:x2)
            Xmatrix, offset = MLJGLMInterface.prepare_inputs(m, X; handle_intercept=true)

            @test offset == [4, 5, 6]
            @test Xmatrix== [1 1;
                             2 1;
                             3 1]
    end

    @testset "no intercept/no offsetcol" for mt in modeltypes
        X = (x1=[1,2,3], x2=[4,5,6])
        m = mt(fit_intercept=false)
        Xmatrix, offset = MLJGLMInterface.prepare_inputs(m, X; handle_intercept=true)

        @test offset == []
        @test Xmatrix == [1 4;
                          2 5;
                          3 6]
    end

end


@testset "Test offsetting models" begin
    @testset "Test split_X_offset" begin
        rng = StableRNGs.StableRNG(123)
        N = 100
        X = (x1=[1,2,3], x2=[4,5,6])
        @test MLJGLMInterface.split_X_offset(X, nothing) == (X, Float64[])
        @test MLJGLMInterface.split_X_offset(X, :x1) == ((x2=[4,5,6],), [1,2,3])

        X = MLJBase.table(rand(rng, N, 3))
        Xnew, offset = MLJGLMInterface.split_X_offset(X, :x2)
        @test offset isa Vector
        @test length(Xnew) == 2
    end

    # In the following:
    # The second column is taken as an offset by the model
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
        @test iszero(fp.intercept)
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

@testset "Param names in fitresult" begin
    X = (a=[1, 9, 4, 2], b=[1, 2, 1, 4], c=[9, 1, 5, 3])
    y = categorical([true, true, false, false])
    lr = LinearBinaryClassifier(fit_intercept=true)
    fitresult, _, report = fit(lr, 1, X, y)
    ctable = last(report)
    parameters = ctable.rownms # Row names.
    @test parameters == ["a", "b", "c", "(Intercept)"]
    intercept = ctable.cols[1][4]
    yhat = predict(lr, fitresult, X)
    @test mean(cross_entropy(yhat, y)) < 0.6

    fp = fitted_params(lr, fitresult)
    @test fp.features == [:a, :b, :c]
    @test :intercept in keys(fp)
    @test intercept == fp.intercept
end

@testset "Param names in report" begin
    X = (a=[1, 4, 3, 1], b=[2, 0, 1, 4], c=[7, 1, 7, 3])
    y = categorical([true, false, true, false])
    # check that by default all possible keys are added in the report
    lr = LinearBinaryClassifier()
    _, _, report = fit(lr, 1, X, y)
    @test :deviance in keys(report) 
    @test :dof_residual in keys(report)
    @test :stderror in keys(report)
    @test :vcov in keys(report)
    @test :coef_table in keys(report)

    # check that report is valid if only some keys are specified
    lr = LinearBinaryClassifier(report_keys = [:stderror, :deviance])
    _, _, report = fit(lr, 1, X, y)
    @test :deviance in keys(report) 
    @test :stderror in keys(report)
    @test :dof_residual ∉ keys(report)

    # check that an empty `NamedTuple` is outputed for
    # `report_params === nothing`
    lr = LinearBinaryClassifier(report_keys=nothing)
    _, _, report = fit(lr, 1, X, y)
    @test report === NamedTuple()
end
