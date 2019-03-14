module InformedDifferentialEvolution

using FunctionalData, FunctionalDataUtils, Random

export DE

extract(a, field) = map(x->x[field],a)
nanmean(a) = mean(a[!isnan(a)])

function rounder!(a,stepsize::Real)
    for i = 1:length(a)
        a[i] = round(a[i]/stepsize)*stepsize
    end
end

function rounder!(a,stepsize::Matrix)
    for n = 1:len(a), m = 1:sizem(a)
        a[m,n] = round(a[m,n]/stepsize[m])*stepsize[m]
    end
end


DE(costf::Function, mi::Vector, ma::Vector; args...) = DE(costf, col(mi), col(ma); args...)
function DE(costf::Function, mi::Array{T,2}, ma::Array{T,2};
    npop = 100,
    maxiter::Int = 1_000_000,
	maxstableiter::Int = 100,
    predictors::Array = Any[:default],
    sampler = rand,
    initpop = if sampler == rand
        col(mi) .+ rand(length(mi), npop) .* (col(ma) - col(mi))
    elseif sampler == randn
        (col(mi)+col(ma))/2 .+ clamp(randn(length(mi), npop)./6,-0.5,0.5) .* (col(ma) - col(mi))
    else
        error("InformedDifferentialEvolution: de: invalid value for paramter 'sampler'")
    end,
    recordhistory = false,
    tryallpredictors = false,
    continueabove = -Inf,
    crossoverprob = 0.5,
    diffweight = 0.85,
	roundto = 1e-6,
    data = nothing,
    verbosity::Array = Symbol[],  # :iter, :best, :newbest, :pop, :newbestcost, :initbestcost
    replaceworst = 0.1,
    classicmode = true,
    passpreviouscost = false,
    # :iter show iter number, :newbest show new bests, :pop population
    io = stdout,
    evaluator = nothing  # called as evaluator(pop, costs, bestind, best), can return arbitrary things for history
   ) where T

    pop = copy(initpop)
    rounder!(pop, roundto)
    clamp!(pop,mi,ma)
    costs = zeros(npop)
    newcosts = zeros(npop)
    frompredictor = falses(len(predictors), npop)
    ind1 = randperm(npop)
    ind2 = randperm(npop)
    iter = 1
    ncostevals = 0
	nstableiter = 0
	if isa(roundto, Array) roundto = col(roundto) end

    ###### predictors

    function defaultpredictor!(r::Matrix{T}, pop::Matrix{T}, costs) where T
        if classicmode
            randind() = rand(1:npop)
            for ind0 = 1:npop
                D = rand(1:sizem(pop))

                ind1 = randind()
                while ind1 == ind0 ind1 = randind() end
                ind2 = randind()
                while ind2 == ind0 || ind2 == ind1 ind2 = randind() end
                ind3 = randind()
                while ind3 == ind0 || ind3 == ind1 || ind3 == ind2 ind3 = randind() end

                for m = 1:sizem(pop)
                    if rand() < crossoverprob || m == D
                        r[m,ind0] = pop[m,ind1] + diffweight*(pop[m,ind2]-pop[m,ind3])
                    else
                        r[m,ind0] = pop[m,ind0]
                    end
                end
            end
            r
        else
            for n = 1:len(pop), m = 1:sizem(pop)
                ind1 = rand(1:npop)
                ind2 = rand(1:npop)
                r[m,n] = pop[m,n] + diffweight*(pop[m,ind1]-pop[m,ind2])
            end
            if replaceworst > 0.0
                # @show costs
                ind = sortperm(costs)
                n = round(length(costs)*replaceworst)
                r[:,ind[end-n+1:end]] = r[:, ind[1:n]]
            end
            r
        end
    end

    if isempty(predictors)
         error("no predictors specified, use [:default] for no predictors")
    end
    predictors = convert(Array{Any}, predictors)
    predictors[findall(predictors.==:default)] .= defaultpredictor!
    @assert all(x->isa(x, Function), predictors)  "InformedDifferentialEvolution: all predictors need to be functions or the symbol :default"
    predictedpop = zero(initpop)

    if !isnil(data)
        costf_(a) = costf(a, data)
    else
        costf_ = costf
    end

    for i = 1:npop
        costs[i] = costf_(pop[:,i])
        ncostevals += 1
    end
    bestcost, bestind = findmin(costs)
    best = pop[:,bestind]

    showiter = in(:iter, verbosity)
    showbest = in(:best, verbosity)
    shownewbest = in(:newbest, verbosity)
    shownewbestcost = in(:newbestcost, verbosity)
    showinitbestcost = in(:initbestcost, verbosity)
    showpop = in(:pop, verbosity)
    log(a...) = println(io, a...)

    if recordhistory
        history = Array(Any, maxiter+1)
        history[1] = Dict(:pop => copy(pop), :costs => copy(costs), :bestcost => copy(best), :bestind => bestind, :frompredictor => copy(frompredictor), :ncostevals => copy(ncostevals), :bestind => indmin(costs))
    else
        history = Any[]
    end


    # pv = FD.view(predictedpop)
    while iter <= maxiter && nstableiter < maxstableiter && bestcost > continueabove
        showinitbestcost && iter == 1 && log("Best cost in init pop: $bestcost")
        showiter && log("Iteration: $iter")
        showpop && log("Population:\n$pop")

		nstableiter += 1

        gotbetter = false
        for m = 1:len(predictors)
            predictors[m](predictedpop, pop, costs)
            rounder!(predictedpop, roundto)
            clamp!(predictedpop, mi, ma)

            for n = 1:npop
                # FD.view!(predictedpop, n, pv)
                pv = at(predictedpop,n)
                if passpreviouscost
                    newcost = costf_(pv, costs[n])
                else
                    newcost = costf_(pv)
                end
                ncostevals += 1
                # @show costs[n] newcost pv
                if newcost < costs[n]
                    frompredictor[m,n] = true
                    pop[:,n] = pv
                    costs[n] = newcost
                end
            end
        end
        # println("^ gotbetter")
        oldbestcost = bestcost
        bestcost, bestind = findmin(costs)
        best = col(pop[:,bestind])

		if bestcost < oldbestcost
            shownewbestcost && log("New best at iter $iter with cost $bestcost")
            shownewbest && log("New best at iter $iter with cost $bestcost:\n", best)
			nstableiter = 0
		end

        if recordhistory
            history[iter+1] = Dict(:pop => copy(pop), :costs => copy(costs), :bestcost => copy(best), :frompredictor => copy(frompredictor), :ncostevals => copy(ncostevals), :bestind => bestind)
            if evaluator != nothing
                history[iter+1][:evaluator] = evaluator(pop, costs, bestind, best)
            end
        end
        iter += 1
    end
    if recordhistory
        history = history[1:iter]
        k = collect(keys(first(history)))
        history = Dict(zip(k, map(x->extract(history, x), k)))
    end
    showbest && log("Best:\n$best")
    best, Dict(:bestcost => bestcost, :ncostevals => ncostevals, :history => history, :pop => pop)
end


end # module InformedDifferentialEvolution

module InformedDEHelpers

using InformedDifferentialEvolution
export analyze, singleanalysis

function interpval(values, ticks, at)
    ind = findall(at.>=ticks)
    if isempty(ind)
        ind = 1
    else
        ind = last(ind)
    end
    #@show ind
    if at==ticks[ind]
        r = values[ind]
    elseif at>ticks[end]
        r = NaN
    else
        #@show values ticks at ind
        mixfactor = (at-ticks[ind])/(ticks[ind+1]-ticks[ind])
        r = mixfactor*values[ind+1] + (1-mixfactor)*values[ind]
    end
    #@show r
    r
end
@assert interpval(1:10, 1:10, 1.5) == 1.5
@assert interpval(-1:-1:-10, 1:10, 8.5) == -8.5
@assert map(x->interpval((1:3).^2,1:3,x),1:3) == [1,4,9]
 
function singleanalysis(f, mi, ma; nruns = 10, npop = 100, maxiter = 100, stepsize = 100, predictors = Any[], tryallpredictors = false, kargs...) 
    nsamples = int((maxiter + 1)*max(1,length(predictors))*npop / stepsize)
    #@show nsamples
    costs = fill(NaN,nsamples)
    counters = zeros(nsamples)
    allbests = Any[]
    allstats = Any[]
    for i = 1:nruns
        best, stats = DE(f, mi, ma, 
        tryallpredictors = tryallpredictors, recordhistory = true, maxiter = maxiter, npop = npop, predictors = predictors; kargs...)
        #@show stats[:bestcost] stats[:ncostevals]
        #@show stats[:history][:ncostevals]
        maxval = stats[:history][:ncostevals][end]
        for j = 1:maxval/stepsize
            #@show length(stats[:history][:bestcost])
            #@show length(stats[:history][:ncostevals])
            #@show j*stepsize
            isnan(costs[j]) ? costs[j] = 0 : nothing
            costs[j] += interpval(stats[:history][:bestcost], stats[:history][:ncostevals], j*stepsize)
            counters[j] += 1
        end
        push!(allbests, best)
        push!(allstats, stats)
    end
    ind = counters.>0
    costs[ind] ./= counters[ind]

    lines = semilogy(stepsize*(1:length(costs)),costs)
    xlabel("Number of cost function evaluations")
    ylabel("Cost of best idividuum")
    #lines = semilogy(stats[:history][:ncostevals], map(nanmean,stats[:history][:bestcost]))
    allbests, allstats, lines[1]
end

function analyze(f, mi, ma, predictors)
    r = map(x->singleanalysis(f, mi, ma, predictors = x[2]; x[3]...), predictors)
    legend(map(x->x[3],r), map(first, predictors))
end

function demo()
	f(x) = sum(abs.(x))
	predictor(pop,costs) = vcat(mean(pop,1), mean(pop,1))
	mi = [-1,-1]; ma = [1,1]

	analyze(f, mi, ma, [
		("default", [:default],Any[]),
		("pred first", [predictor, :default], Any[]),
		("pred last", [:default, predictor], Any[]),
		("pred only", [predictor], Any[]),
		("tryall", [predictor, :default], [(:tryallpredictors,true)])
	])
end

end # module
