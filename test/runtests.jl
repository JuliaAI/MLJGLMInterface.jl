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
### END-TO-END TESTS
###
@testset "OLSREGRESSOR" begin
    X, y = @load_boston

    train, test = partition(eachindex(y), 0.7)

    atom_ols = LinearRegressor()

    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest  = selectrows(X, test)

    # Without weights `w`
    fitresult, _, _ = fit(atom_ols, 1, Xtrain, ytrain)
    θ = MLJBase.fitted_params(atom_ols, fitresult)

    p = predict_mean(atom_ols, fitresult, Xtest)

    # With weihts `w`
    fitresultw, _, _ = fit(atom_ols, 1, Xtrain, ytrain, ones(eltype(ytrain), length(ytrain)))
    θw = MLJBase.fitted_params(atom_ols, fitresult)

    pw = predict_mean(atom_ols, fitresultw, Xtest)

    # hand made regression to compare

    Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
    Xa1 = hcat(Xa, ones(size(Xa, 1)))
    coefs = Xa1[train, :] \ y[train]
    p2 = Xa1[test, :] * coefs

    @test p ≈ p2
    @test pw ≈ p2

    p_distr = predict(atom_ols, fitresult, selectrows(X, test))

    @test p_distr[1] == Distributions.Normal(
        p[1], MLJGLMInterface.dispersion(fitresult))

    # check predict on `Xtest1` with wrong dims
    Xtest1 = MLJBase.table(Tables.matrix(Xtest)[:, 1:3], names=Tables.columnnames(Xtest)[1:3])
    @test_throws DimensionMismatch predict(atom_ols, fitresult, Xtest1)
    # check exact match of features during predict
    atom_ols.check_features = true
    Xtest2 = MLJBase.table(
        Tables.matrix(Xtest), names=[Symbol("p$i") for i in eachindex(Tables.columnnames(Xtest))]
    )
    @test_throws ArgumentError predict(atom_ols, fitresult, Xtest2)

    # Test metadata
    model = atom_ols
    @test name(model) == "LinearRegressor"
    @test package_name(model) == "GLM"
    @test supports_weights(model)
    @test is_pure_julia(model)
    @test is_supervised(model)
    @test package_license(model) == "MIT"
    @test prediction_type(model) == :probabilistic
    @test hyperparameters(model) == (
        :fit_intercept, :dropcollinear, :offsetcol, :report_params, :check_features
    )
    @test hyperparameter_types(model) == (
        "Bool",
        "Bool",
        "Union{Nothing, Symbol}",
        "Union{Nothing, AbstractVector{Symbol}}",
        "Bool"
    )

end

@testset "Logistic regression" begin
    rng = StableRNGs.StableRNG(0)

    N = 100
    X = MLJBase.table(rand(rng, N, 4));
    ycont = 2*X.x1 - X.x3 + 0.6*rand(rng, N)
    y = categorical(ycont .> mean(ycont))

    lr = LinearBinaryClassifier()
    fitresult, _, report = fit(lr, 1, X, y)

    yhat = predict(lr, fitresult, X)
    @test mean(cross_entropy(yhat, y)) < 0.25

    pr = LinearBinaryClassifier(link=GLM.ProbitLink())

    # Without weights
    fitresult, _, report = fit(pr, 1, X, y)
    yhat = predict(lr, fitresult, X)
    @test mean(cross_entropy(yhat, y)) < 0.25

    # With weights
    fitresultw, _, reportw = fit(pr, 1, X, y)
    yhatw = predict(lr, fitresultw, X)
    @test mean(cross_entropy(yhatw, y)) < 0.25
    @test yhatw ≈ yhat

    fitted_params(pr, fitresult)

    # check predict on `Xnew` with wrong dims
    Xnew = MLJBase.table(Tables.matrix(X)[:, 1:3], names=Tables.columnnames(X)[1:3])
    @test_throws DimensionMismatch predict(lr, fitresult, Xnew)
    # check exact match of features during predict
    lr.check_features = true
    Xnew1 = MLJBase.table(
        Tables.matrix(X), names=[Symbol("p$i") for i in eachindex(Tables.columnnames(X))]
    )
    @test_throws ArgumentError predict(lr, fitresult, Xnew1)

    # Test metadata
    model = lr
    @test name(model) == "LinearBinaryClassifier"
    @test package_name(model) == "GLM"
    @test supports_weights(model)
    @test is_pure_julia(model)
    @test is_supervised(model)
    @test package_license(model) == "MIT"
    @test prediction_type(model) == :probabilistic
    @test hyperparameters(model) == (
        :fit_intercept,
        :link, 
        :offsetcol,
        :maxiter,
        :atol,
        :rtol,
        :minstepfac,
        :report_params,
        :check_features
    )
    @test hyperparameter_types(model) == (
        "Bool",
        "$(GLM.Link01)",
        "Union{Nothing, Symbol}",
        "Integer",
        "Real",
        "Real",
        "Real",
        "Union{Nothing, AbstractVector{Symbol}}",
        "Bool"
    )
end

@testset "Count regression" begin
    rng = StableRNGs.StableRNG(123)

    X = randn(rng, 500, 5)
    θ = randn(rng, 5)
    y = map(exp.(X*θ)) do mu
        rand(rng, Distributions.Poisson(mu))
    end

    XTable = MLJBase.table(X)

    lcr = LinearCountRegressor(fit_intercept=false)

    # Without weights
    fitresult, _, _ = fit(lcr, 1, XTable, y)
    θ̂ = fitted_params(lcr, fitresult).coef
    @test norm(θ̂ .- θ)/norm(θ) ≤ 0.03

    # With weights
    fitresultw, _, _ = fit(lcr, 1, XTable, y)
    θ̂w = fitted_params(lcr, fitresultw).coef
    @test norm(θ̂w .- θ)/norm(θ) ≤ 0.03

    # check predict on `Xnew` with wrong dims
    Xnew = MLJBase.table(Tables.matrix(XTable)[:, 1:3], names=Tables.columnnames(XTable)[1:3])
    @test_throws DimensionMismatch predict(lcr, fitresult, Xnew)
    # check exact match of features during predict
    lcr.check_features = true
    Xnew1 = MLJBase.table(
        Tables.matrix(XTable),
        names=[Symbol("p$i") for i in eachindex(Tables.columnnames(XTable))]
    )
    @test_throws ArgumentError predict(lcr, fitresult, Xnew1)

    # Test metadata
    model = lcr
    @test name(model) == "LinearCountRegressor"
    @test package_name(model) == "GLM"
    @test supports_weights(model)
    @test is_pure_julia(model)
    @test is_supervised(model)
    @test package_license(model) == "MIT"
    @test prediction_type(model) == :probabilistic
    @test hyperparameters(model) == (
        :fit_intercept,
        :distribution,
        :link, 
        :offsetcol,
        :maxiter,
        :atol,
        :rtol,
        :minstepfac,
        :report_params,
        :check_features
    )
    @test hyperparameter_types(model) == (
        "Bool",
        "$(Distributions.Distribution)",
        "$(GLM.Link)",
        "Union{Nothing, Symbol}",
        "Integer",
        "Real",
        "Real",
        "Real",
        "Union{Nothing, AbstractVector{Symbol}}",
        "Bool"
    )

end


###
### Unit tests
###
const modeltypes = [LinearRegressor, LinearBinaryClassifier, LinearCountRegressor]
@testset "Test prepare_inputs" begin
    @testset "intercept/offsetcol" for mt in modeltypes
            X = (x1=[1,2,3], x2=[4,5,6])
            m = mt(fit_intercept=true, offsetcol=:x2)
            Xmatrix, offset = MLJGLMInterface.prepare_inputs(m, X)

            @test offset == [4, 5, 6]
            @test Xmatrix== [
                1 1;
                2 1;
                3 1
            ]
            # try different tables
            X2 = Tables.table(Tables.matrix(X), header=Tables.columnnames(X))::Tables.MatrixTable
            Xmatrix2, offset2 = MLJGLMInterface.prepare_inputs(m, X2)
            @test offset2 == offset
            @test Xmatrix2 == Xmatrix

            X3 = Tables.Columns(X)::Tables.Columns
            Xmatrix3, offset3 = MLJGLMInterface.prepare_inputs(m, X3)
            @test offset3 == offset
            @test Xmatrix3 == Xmatrix
            
            # throw error when offsetcol isn't present in table
            m1 = mt(fit_intercept=true, offsetcol=:x3)
            @test_throws ArgumentError MLJGLMInterface.prepare_inputs(m1, X)
    end

    @testset "no intercept/no offsetcol" for mt in modeltypes
        X = (x1=[1,2,3], x2=[4,5,6])
        m = mt(fit_intercept=false)
        Xmatrix, offset = MLJGLMInterface.prepare_inputs(m, X)

        @test offset == []
        @test Xmatrix == [
            1 4;
            2 5;
            3 6
        ]

        # throw error when fitting table with no columns `e.g NamedTuple()`
        X1 = NamedTuple()
        @test_throws ArgumentError MLJGLMInterface.prepare_inputs(m, X1)
        
    end

    @testset "offsetcol but no intercept" for mt in modeltypes
        X = (x1=[1,2,3], x2=[4,5,6])
        m = mt(offsetcol=:x1, fit_intercept=false)
        Xmatrix, offset = MLJGLMInterface.prepare_inputs(m, X)

        @test offset == [1, 2, 3]
        @test Xmatrix == permutedims([4 5 6])

        # throw error for tables with just one column and for which 
        # `offsetcol !== nothing` and `fit_intercept == false`
        # as at least two columns would be required in this case.
        X1 = (x1=[1,2,3],)
        @test_throws ArgumentError MLJGLMInterface.prepare_inputs(m, X1)
    end
end

@testset "Test offsetting models" begin
    @testset "Test split_X_offset" begin
        rng = StableRNGs.StableRNG(123)
        N = 100
        X = (x1=[1,2,3], x2=[4,5,6])
        @test MLJGLMInterface.split_X_offset(X, nothing) == (X, Float64[], [:x1, :x2])
        @test MLJGLMInterface.split_X_offset(X, :x1) == ((x2=[4,5,6],), [1,2,3], [:x2,])

        X = MLJBase.table(rand(rng, N, 3))
        Xnew, offset, X_features = MLJGLMInterface.split_X_offset(X, :x2)
        @test offset isa Vector
        @test length(Xnew) == 2
        @test X_features isa Vector{Symbol}
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

@testset "Param names in fitresult/report" begin
    X = (a=[1, 9, 4, 2], b=[1, 2, 1, 4], c=[9, 1, 5, 3])
    y = categorical([true, true, false, false])
    lr = LinearBinaryClassifier(fit_intercept=true)
    fitresult, _, report = fit(lr, 1, X, y)
    # by default `report` contains every training stats about the fit
    @test :deviance in keys(report) 
    @test :dof_residual in keys(report)
    @test :stderror in keys(report)
    @test :vcov in keys(report)
    @test :coef_table in keys(report)
    ctable = last(report)
    parameters = ctable.rownms # Row names.
    @test parameters == ["a", "b", "c", "(Intercept)"]
    intercept = ctable.cols[1][4]
    coef = ctable.cols[1][1:3]
    yhat = predict(lr, fitresult, X)
    @test mean(cross_entropy(yhat, y)) < 0.6

    fp = fitted_params(lr, fitresult)
    @test :coef in keys(fp)
    @test fp.coef == coef
    @test :feature_names in keys(fp)
    @test fp.feature_names == [:a, :b, :c]
    @test :intercept in keys(fp)
    @test intercept == fp.intercept

    # test when `fit_intercept` is false
    lr = LinearBinaryClassifier(fit_intercept=false)
    fitresult, _, report = fit(lr, 1, X, y)
    ctable = last(report)
    parameters = ctable.rownms # Row names.
    @test parameters == ["a", "b", "c"]
    coef = ctable.cols[1][1:3]
    fp = fitted_params(lr, fitresult)
    @test :coef in keys(fp)
    @test fp.coef == coef
    @test :feature_names in keys(fp)
    @test fp.feature_names == [:a, :b, :c]
    @test :intercept in keys(fp)
    @test fp.intercept == 0

end
