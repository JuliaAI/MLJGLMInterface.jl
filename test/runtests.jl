using Test

using MLJBase
using StatisticalMeasures
using LinearAlgebra
using Statistics
using MLJGLMInterface
using GLM: coeftable
import GLM

using  Distributions: Normal, Poisson, Uniform
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

    fitresult, _, _ = fit(atom_ols, 1, Xtrain, ytrain)
    θ = MLJBase.fitted_params(atom_ols, fitresult)

    p = predict_mean(atom_ols, fitresult, Xtest)

    fitresultw, _, _ = fit(atom_ols, 1, Xtrain, ytrain, wtrain)
    θw = MLJBase.fitted_params(atom_ols, fitresult)

    pw = predict_mean(atom_ols, fitresultw, Xtest)

    Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
    Xa1 = hcat(Xa, ones(size(Xa, 1)))
    coefs = Xa1[train, :] \ y[train]
    p2 = Xa1[test, :] * coefs

    @test p ≈ p2
    @test pw ≈ p2

    # test `predict` object
    p_distr = predict(atom_ols, fitresult, selectrows(X, test))
    dispersion =  MLJGLMInterface.dispersion(fitresult)
    @test p_distr[1] == Normal(p[1], dispersion)

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

    fitresult, _, report = fit(lr, 1, X, y)
    yhat = predict(lr, fitresult, X)
    @test cross_entropy(yhat, y) < 0.25
    fitresult1, _, report1 = fit(pr, 1, X, y)
    yhat1 = predict(pr, fitresult1, X)
    @test cross_entropy(yhat1, y) < 0.25

    fitresultw, _, reportw = fit(lr, 1, X, y, w)
    yhatw = predict(lr, fitresultw, X)
    @test cross_entropy(yhatw, y) < 0.25
    @test yhatw ≈ yhat
    fitresultw1, _, reportw1 = fit(pr, 1, X, y, w)
    yhatw1 = predict(pr, fitresultw1, X)
    @test cross_entropy(yhatw1, y) < 0.25
    @test yhatw1 ≈ yhat1

    # check predict on `Xnew` with wrong dims
    Xnew = MLJBase.table(Tables.matrix(X)[:, 1:3], names=Tables.columnnames(X)[1:3])
    @test_throws DimensionMismatch predict(lr, fitresult, Xnew)

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
    hyper_params = hyperparameters(model)
    @test hyper_params[1] == :fit_intercept
    @test hyper_params[2] == :link
    @test hyper_params[3] == :offsetcol
    @test hyper_params[4] == :maxiter
    @test hyper_params[5] == :atol
    @test hyper_params[6] == :rtol
    @test hyper_params[7] == :minstepfac
    @test hyper_params[8] == :report_keys

end

###
### Count regression
###
@testset "Count regression" begin
    rng = StableRNGs.StableRNG(123)

    X = randn(rng, 500, 5)
    θ = randn(rng, 5)
    y = map(exp.(X*θ)) do mu
        rand(rng, Poisson(mu))
    end
    w = ones(eltype(y), length(y))

    XTable = MLJBase.table(X)

    lcr = LinearCountRegressor(fit_intercept=false)

    fitresult, _, _ = fit(lcr, 1, XTable, y)
    θ̂ = fitted_params(lcr, fitresult).coef
    @test norm(θ̂ .- θ)/norm(θ) ≤ 0.03

    fitresultw, _, _ = fit(lcr, 1, XTable, y, w)
    θ̂w = fitted_params(lcr, fitresultw).coef
    @test norm(θ̂w .- θ)/norm(θ) ≤ 0.03
    @test θ̂w ≈ θ̂

    # check predict on `Xnew` with wrong dims
    Xnew = MLJBase.table(
        Tables.matrix(XTable)[:, 1:3], names=Tables.columnnames(XTable)[1:3]
    )
    @test_throws DimensionMismatch predict(lcr, fitresult, Xnew)

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
    @test hyper_params[5] == :maxiter
    @test hyper_params[6] == :atol
    @test hyper_params[7] == :rtol
    @test hyper_params[8] == :minstepfac
end

modeltypes = [LinearRegressor, LinearBinaryClassifier, LinearCountRegressor]
@testset "Test prepare_inputs" begin
    @testset "check sample size" for fit_intercept in [true, false]
        lin_reg = LinearRegressor(; fit_intercept)
        X_lin_reg = MLJBase.table(rand(3, 3 + fit_intercept))
        log_reg = LinearBinaryClassifier(; fit_intercept)
        X_log_reg = MLJBase.table(rand(2, 3 + fit_intercept))
        lcr = LinearCountRegressor(; distribution=Poisson(), fit_intercept)
        X_lcr = X_log_reg
        for (m, X) in [(lin_reg, X_lin_reg), (log_reg, X_log_reg), (lcr, X_lcr)]
            @test_throws ArgumentError MLJGLMInterface.prepare_inputs(m, X)
        end

    end

    @testset "intercept/offsetcol" for mt in modeltypes
            X = (x1=[1,2,3], x2=[4,5,6])
            m = mt(fit_intercept=true, offsetcol=:x2)
            r = MLJGLMInterface.prepare_inputs(m, X; handle_intercept=true)
            Xmatrix, offset, features = r
            @test offset == [4, 5, 6]
            @test Xmatrix== [1 1;
                             2 1;
                             3 1]
            @test features == [:x1]

            m1 = mt(fit_intercept=true, offsetcol=:x3)
            @test_throws ArgumentError MLJGLMInterface.prepare_inputs(m1, X)
    end

    @testset "no intercept/no offsetcol" for mt in modeltypes
        X = (x1=[1,2,3], x2=[4,5,6])
        m = mt(fit_intercept=false)
        r = MLJGLMInterface.prepare_inputs(m, X; handle_intercept=true)
        Xmatrix, offset, features = r
        @test offset == []
        @test Xmatrix == [1 4;
                          2 5;
                          3 6]
        @test features == [:x1, :x2]

        X1 = NamedTuple()
        @test_throws ArgumentError MLJGLMInterface.prepare_inputs(m, X1)
    end

    @testset "offsetcol but no intercept" for mt in modeltypes
        X = (x1=[1,2,3], x2=[4,5,6])
        m = mt(offsetcol=:x1, fit_intercept=false)
        Xmatrix, offset, features = MLJGLMInterface.prepare_inputs(m, X)

        @test offset == [1, 2, 3]
        @test Xmatrix == permutedims([4 5 6])
        @test features == [:x2]

        # throw error for tables with just one column.
        # Since for `offsetcol !== nothing` and `fit_intercept == false`
        # the table must have at least two columns.
        X1 = (x1=[1,2,3],)
        @test_throws ArgumentError MLJGLMInterface.prepare_inputs(m, X1)
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
        y = rand(rng, Uniform(0,1), N) .< expit(2*X.x1 + X.x2 - X.x3)
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
        y = 2*X.x1 + X.x2 - X.x3 + rand(rng, Normal(0,1), N)

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
            rand(rng, Poisson(mu))
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
    @test cross_entropy(yhat, y) < 0.6

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
    @test :raw_glm_model in keys(report)

    @test report.raw_glm_model isa GLM.GeneralizedLinearModel

    # check that report is valid if only some keys are specified
    lr = LinearBinaryClassifier(report_keys = [:stderror, :deviance])
    _, _, report = fit(lr, 1, X, y)
    @test :deviance in keys(report)
    @test :stderror in keys(report)
    @test :dof_residua ∉ keys(report)
    @test :raw_glm_model ∉ keys(report)

    # check that an empty `NamedTuple` is outputed for
    # `report_params === nothing`
    lr = LinearBinaryClassifier(report_keys=nothing)
    _, _, report = fit(lr, 1, X, y)
    @test report === NamedTuple()
end

@testset "Issue 27" begin
    @inferred MLJGLMInterface.augment_X(rand(2, 2), false)
    n, p = (100, 2)
    Xmat = rand(p, n)' # note using adjoint
    X = MLJBase.table(Xmat)
    y = rand(n)
    lr = LinearRegressor()
    # Smoke test whether it crashes on an LinearAlgebra.Adjoint.
    fit(lr, 1, X, y)
end

@testset "Issue 34" begin
    model = LinearRegressor()
    form = MLJGLMInterface.glm_formula(model, [:a, :b]) |> string
    @test form == "y ~ a + b + 1"
end
