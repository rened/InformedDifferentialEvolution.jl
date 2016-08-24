println("\n\n\nRunning tests ...") 

using InformedDifferentialEvolution
using Base.Test

optimum = [-2 4]'
costf(x) = (r = sqrt(sum((x-optimum).^2)); sin(r)^2+r/2)
costf2(x, ignore) = (r = sqrt(sum((x-optimum).^2)); sin(r)^2+r/2)

best, info = de(costf, [-10,-10], [10,10])
@test best == optimum

best, info = de(costf2, [-10,-10], [10,10], data = 1)
@test best == optimum

f(x) = sum(abs(x))
predictor(r,pop,costs) = r[:] = vcat(mean(pop,1), mean(pop,1))
mi = [-1,-1]; ma = [1,1]

best, info = de(f, mi, ma, predictors = [predictor, :default])
@test maximum(abs(best)) < 0.000001

function g(x, previouscost = Inf)
    r = 0.
    for i = -5:5
        r += abs(x[1])+abs(x[2]+i)
        if r > previouscost
            break
        end
    end
    r
end
best, info = de(g, [-100,-100], [100,100])
@test best == [0 0]'

println("   done running tests!")
