module InformedDifferentialEvolution

export de

col(a) = reshape(a, (length(a),1))
function clamp!(a, mi, ma)
   for n = size(a,2), m = size(a,1) 
        a[m,n] = clamp(a[m,n],mi[m],ma[m]) 
    end
end

extract(a, field) = map(x->x[field],a)
nanmean(a) = mean(a[!isnan(a)])

rounder(a,stepsize) = round(a./stepsize).*stepsize

de(costf::Function, mi::Vector, ma::Vector; args...) = de(costf, col(mi), col(ma); args...)
function de{T}(costf::Function, mi::Array{T,2}, ma::Array{T,2}; 
    npop = 100, 
    maxiter = 1e6, 
	maxstableiter = 100,
    predictors = {:default},
    lambda = 0.85,
    initpop = mi .+ rand(length(mi), npop) .* (ma - mi),
    recordhistory = false,
    tryallpredictors = false,
    continueabove = -Inf,
    replaceworst = 0.0,
	roundto = 1e-6)

    pop = copy(initpop)
    newpop = zero(initpop)
    costs = zeros(npop)
    newcosts = zeros(npop)
    frompredictor = fill(NaN,npop)
    ind1 = randperm(npop)
    ind2 = randperm(npop)
    iter = 1
    ncostevals = 0
	nstableiter = 0
	if isa(roundto, Array) roundto = col(roundto) end

    for i = 1:npop
        costs[i] = costf(pop[:,i])
        ncostevals += 1
    end
    bestcost, bestind = findmin(costs)
    best = pop[:,bestind]

    if recordhistory
        history = Array(Any,maxiter+1)
        history[1] = Dict({"pop","costs","bestcost","best","predictor", "ncostevals"},
        {copy(pop), copy(costs), copy(bestcost), copy(best), copy(frompredictor), ncostevals})
    else
        history = {}
    end
        
    while iter < maxiter && nstableiter < maxstableiter && bestcost > continueabove
		nstableiter += 1

        if isempty(predictors)
            error("no predictors specified, use {:default} for no predictors")
        end
        function defaultpredictor(pop, costs)
            r = pop + lambda*(pop[:,randperm(npop)]-pop[:,randperm(npop)])
            if replaceworst > 0.0
                # @show costs
                ind = sortperm(costs)
                n = round(length(costs)*replaceworst)
                r[:,ind[end-n+1:end]] = r[:, ind[1:n]]
            end
            r
        end

        predictors[find(predictors.==:default)] = defaultpredictor
        predictedpops = map(x->x(pop, costs), predictors)
		if roundto != nothing && roundto > 0
			map(x->rounder(x, roundto), predictedpops)
		end
				
                
		gotbetter = false
        for i = 1:npop
            gotbetter = false

            for j = 1:length(predictedpops)
                if !gotbetter || tryallpredictors
                    newitem = predictedpops[j][:,i]
                    newcost = costf(newitem)
                    ncostevals += 1
                    if newcost < costs[i]
                        gotbetter = true
                        frompredictor[i] = j
                        pop[:,i] = newitem
                        costs[i] = newcost
                    end
                end
            end
        end
		if gotbetter
			nstableiter = 0
		end
        bestcost, bestind = findmin(costs)
        best = col(pop[:,bestind])
		# @show bestcost nstableiter
        if recordhistory
            history[iter+1] = Dict({"pop","costs","bestcost","best","predictor","ncostevals"},
            {copy(pop), copy(costs), copy(bestcost), copy(best), copy(frompredictor), copy(ncostevals)})
        end
        iter += 1
    end
    if recordhistory
        history = history[1:iter]
        k = collect(keys(first(history)))
        history = Dict(k, map(x->extract(history, x), k))
    end
    best, Dict(("bestcost", "ncostevals", "history"), (bestcost, ncostevals, history))
end


end # module InformedDifferentialEvolution

module InformedDEHelpers

using InformedDifferentialEvolution, PyPlot
export analyze, singleanalysis

function interpval(values, ticks, at)
    ind = find(at.>=ticks)
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
assert(interpval(1:10, 1:10, 1.5)==1.5)
assert(interpval(-1:-1:-10, 1:10, 8.5)==-8.5)
assert(map(x->interpval((1:3).^2,1:3,x),1:3)==[1,4,9])
 
function singleanalysis(f, mi, ma; nruns = 10, npop = 100, maxiter = 100, stepsize = 100, predictors = {}, tryallpredictors = false, kargs...) 
    nsamples = int((maxiter + 1)*max(1,length(predictors))*npop / stepsize)
    #@show nsamples
    costs = fill(NaN,nsamples)
    counters = zeros(nsamples)
    allbests = {}
    allstats = {}
    for i = 1:nruns
        best, stats = de(f, mi, ma, 
        tryallpredictors = tryallpredictors, recordhistory = true, maxiter = maxiter, npop = npop, predictors = predictors; kargs...)
        #@show stats["bestcost"] stats["ncostevals"]
        #@show stats["history"]["ncostevals"]
        maxval = stats["history"]["ncostevals"][end]
        for j = 1:maxval/stepsize
            #@show length(stats["history"]["bestcost"])
            #@show length(stats["history"]["ncostevals"])
            #@show j*stepsize
            isnan(costs[j]) ? costs[j] = 0 : nothing
            costs[j] += interpval(stats["history"]["bestcost"], stats["history"]["ncostevals"], j*stepsize) 
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
    #lines = semilogy(stats["history"]["ncostevals"], map(nanmean,stats["history"]["bestcost"]))
    allbests, allstats, lines[1]
end

function analyze(f, mi, ma, predictors)
    r = map(x->singleanalysis(f, mi, ma, predictors = x[2]; x[3]...), predictors)
    legend(map(x->x[3],r), map(first, predictors))
end

function demo()
	f(x) = sum(abs(x))
	predictor(pop,costs) = vcat(mean(pop,1), mean(pop,1))
	mi = [-1,-1]; ma = [1,1]

	analyze(f, mi, ma, [
		("default", {:default},{}),
		("pred first", {predictor, :default}, {}),
		("pred last", {:default, predictor}, {}),
		("pred only", {predictor}, {}),
		("tryall", {predictor, :default}, {(:tryallpredictors,true)})
	])
end


end # module
