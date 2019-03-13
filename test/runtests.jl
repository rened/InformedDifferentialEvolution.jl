println("\n\n\nRunning tests ...") 

push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using InformedDifferentialEvolution
using Test, Statistics

optimum = [-2 4]'
costf(x) = (r = sqrt(sum((x-optimum).^2)); sin(r)^2+r/2)
costf2(x, ignore) = (r = sqrt(sum((x-optimum).^2)); sin(r)^2+r/2)

best, info = DE(costf, [-10,-10], [10,10])
@test best == optimum

best, info = DE(costf2, [-10,-10], [10,10], data = 1)
@test best == optimum

f(x) = sum(abs.(x))
predictor(r,pop,costs) = r[:] = vcat(mean(pop,dims=1), mean(pop,dims=1))
mi = [-1,-1]; ma = [1,1]

best, info = DE(f, mi, ma, predictors = [predictor, :default])
@test maximum(abs.(best)) < 0.000001

function g(x, previouscost = Inf)
    r = 0.
    for i = -5:5
        r += abs.(x[1])+abs.(x[2]+i)
        if r > previouscost
            break
        end
    end
    r
end
best, info = DE(g, [-100,-100], [100,100])
@test best == [0 0]'

println("   done running tests!")
