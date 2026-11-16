#!/usr/bin/env julia
# SAMOS: frozen K-only event HMMs + causal per-minute K/Z CNN.
# Eight residual blocks; full 769-minute receptive field (12 h 49 min).
# Leaky ReLUs in hidden layers; final output is a raw logit (BCE-with-logits).
# Requires 04_k_shape_mixture.jl and 05_oos_k_mixture.jl in this directory.
# Julia packages: NCDatasets, CSV, Flux.
#
# 1. Fit FINAL frozen K-only HMMs with 04_k_shape_mixture.jl (TRAIN YEARS ONLY).
# 2. Test: julia --threads=1 --gcthreads=1 06_samos_event_cnn.jl --self-test=true
# 3. Prepare: ... 06_samos_event_cnn.jl --prepare-only=true
# 4. Train hybrid: ... 06_samos_event_cnn.jl --mode=hybrid --epochs=10
# 5. Train control: ... 06_samos_event_cnn.jl --mode=raw --epochs=10
# 6. One-time frozen test: ... --mode=hybrid --test-only=true
#
# HMMs are trained ONLY on K runs. CNN uses K/Z T flags as labels, never as inputs.
# Inference uses ONLY measurements available up to the target minute.
# Other T QC flags are excluded from the CNN loss and evaluation; their raw
# measurements remain contextual inputs. Confirm that policy with Kristen.
# Each ship has a chronological train / validation / untouched-test split.

include(joinpath(@__DIR__, "05_oos_k_mixture.jl"))
using Flux

# 5-tap causal stem + two 3-tap causal convolutions per residual block.
# Repeating dilation 64 is intentional: sum(dilations)=191, RF=5+4*191=769.
# With a 1,440-minute day, the first prediction can see the previous day's
# final 768 minutes when the daily files are consecutive.
const FUSION_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 64)
const FUSION_RF = 5 + 4 * sum(FUSION_DILATIONS)
const FUSION_HIST = FUSION_RF
const FUSION_VERSION = "samos_event_cnn_causal_rf769_leaky_v2"
const FUSION_DAY = 1440
const FUSION_RAW = 15
const FUSION_EVENT = 13
const FUSION_CHANNELS = FUSION_RAW + FUSION_EVENT
const FUSION_SHIPLIST = ("WTEE", "WTEP", "WTDH", "ZGOJ7", "ZCYL5")

function fusion_options()
    o = Dict(
        "root"=>"selected-data",
        "exclude-csv"=>"kristen_periods_for_review.csv",
        "hmm"=>"results/final_k_events/k_shape_mixture.jls",
        "output"=>"results/samos_event_cnn_rf769_leaky",
        "cache-dir"=>"results/event_cnn_cache",
        "ships"=>"WTEE,WTEP,WTDH,ZGOJ7",
        "mode"=>"hybrid", "width"=>"24", "epochs"=>"10", "patience"=>"3",
        "lr"=>"0.0003", "dropout"=>"0.05", "pos-weight-power"=>"0.5",
        "target-fpr"=>"0.05", "max-days"=>"0", "prepare-only"=>"false",
        "self-test"=>"false", "rebuild-cache"=>"false",
        "test"=>"false", "test-only"=>"false", "seed"=>"2026", "write-predictions"=>"false"
    )
    for a in ARGS
        startswith(a,"--") && occursin('=',a) || error("Expected --key=value: $a")
        k,v = split(a[3:end],'=';limit=2)
        haskey(o,k) || error("Unknown option --$k")
        o[k]=v
    end
    for k in ("width","epochs","patience")
        parse(Int,o[k])>0 || error("--$k must be positive")
    end
    parse(Int,o["max-days"])>=0 || error("--max-days must be >= 0")
    for k in ("prepare-only","self-test","rebuild-cache","test","test-only","write-predictions")
        lowercase(o[k]) in ("true","false") || error("--$k must be true/false")
    end
    o["mode"] in ("hybrid","raw") || error("--mode=hybrid or --mode=raw")
    0<parse(Float64,o["target-fpr"])<1 || error("Bad target FPR")
    0<parse(Float64,o["pos-weight-power"])<=1 || error("Bad pos-weight-power")
    ships=strip.(split(o["ships"],','))
    all(s->s in FUSION_SHIPLIST, ships) || error("Unknown vessel in --ships")
    length(unique(ships))==length(ships) || error("Duplicate vessel")
    return o,ships
end
fusion_yes(o,k)=lowercase(o[k])=="true"

function fusion_split(ship,yr)
    if ship=="ZCYL5"
        return yr<=2019 ? :train : yr==2020 ? :val : yr==2021 ? :test : :ignore
    elseif ship=="WTEE"
        return yr<=2022 ? :train : yr==2023 ? :val : yr==2025 ? :test : :ignore
    end
    return yr<=2023 ? :train : yr==2024 ? :val : yr==2025 ? :test : :ignore
end

function fusion_files(root,ship)
    found=Dict{Date,String}()
    for folder in (joinpath(root,ship),joinpath(root,ship*"-research-q"))
        isdir(folder) || continue
        for (dir,_,names) in walkdir(folder), name in names
            m=match(r"_(\d{8})v3\d+\.nc$"i,name)
            m===nothing && continue
            day=try Date(m.captures[1],dateformat"yyyymmdd") catch; continue end
            fusion_split(ship,year(day))==:ignore && continue
            path=joinpath(dir,name)
            if !haskey(found,day) || basename(path)>basename(found[day])
                found[day]=path
            end
        end
    end
    found
end

struct FusionDay
    t::Vector{Float32}
    rh::Vector{Float32}
    y::Vector{Int8}
end

# Read T, RH and PRIMARY T QC flags only. Do not use RH's own QC flags as labels.
# Numeric time is read in its native minutes-since-1980 representation.
function fusion_read_day(path,day)
    t=fill(NaN32,FUSION_DAY)
    rh=fill(NaN32,FUSION_DAY)
    y=fill(Int8(-1),FUSION_DAY)
    NCDataset(path,"r") do ds
        (haskey(ds,"time") && haskey(ds,"T")) || error("Missing time or T")
        qc=tflags(ds)
        qc===nothing && error("Missing primary-T QC mapping")
        units=lowercase(strip(string(get(ds["time"].attrib,"units",""))))
        startswith(units,"minutes since 1-1-1980") || error("Unknown time units: $units")
        times=vec(Array(ds["time"].var))
        ref=Dates.value(DateTime(day)-DateTime(1980,1,1))/60000
        locations=zeros(Int,length(times))
        for i in eachindex(times)
            ismissing(times[i]) && continue
            v=Float64(times[i])-ref+1
            isfinite(v) || continue
            j=round(Int,v)
            1<=j<=FUSION_DAY && abs(v-j)<0.05 && (locations[i]=j)
        end
        for (c, name) in enumerate(("T","RH"))
            haskey(ds,name) || continue
            vals=vec(Array(ds[name]))
            uu=lowercase(strip(string(get(ds[name].attrib,"units",""))))
            kelvin=c==1 && (occursin("kelvin",uu) || uu=="k")
            dest=c==1 ? t : rh
            for i in 1:min(length(vals),length(locations))
                j=locations[i]
                (j>0 && !ismissing(vals[i])) || continue
                v=Float32(vals[i]); isfinite(v) || continue
                kelvin && (v-=273.15f0)
                if c==1 ? (-120f0<v<150f0) : (0f0<=v<=100f0)
                    dest[j]=v
                end
            end
        end
        for i in 1:min(length(qc),length(locations))
            j=locations[i]
            j==0 && continue
            if isfinite(t[j])
                y[j] = qc[i]=='K' ? Int8(1) : qc[i]=='Z' ? Int8(0) : Int8(-1)
            end
        end
    end
    return FusionDay(t,rh,y)
end

@inline function fusion_dewpoint(t::Float64,rh::Float64)
    # Approximate Magnus dew point above liquid water. Do not treat as a hard QC test.
    if !(-50<t<60 && 0<rh<=100)
        return NaN
    end
    a=17.625; b=243.04
    g=log(rh/100)+a*t/(b+t)
    return b*g/(a-g)
end

mutable struct FusionTracker
    tr::Ring
    hr::Ring
    state::Matrix{Float64}    # 2 x components, HMM filtered probabilities
    initialized::Bool
    ema15::Float64
    ema60::Float64
    models::Vector{GHMM}
    weights::Vector{Float64}
    center::Vector{Float64}
    scale::Vector{Float64}
    fw::Vector{Float64}
    rawfeat::Vector{Float64}
    z::Vector{Float64}
end

function FusionTracker(bundle)
    models=bundle.models
    all(length(m.pi)==2 for m in models) || error("This encoder expects two states per component")
    length(models)==3 || error("Expected three mixture HMMs")
    bundle.feature_names==FNAMES || error("HMM checkpoint feature definition changed")
    return FusionTracker(Ring(),Ring(),zeros(2,length(models)),false,0,0,
                         models,Float64.(bundle.weights),Float64.(bundle.center),
                         Float64.(bundle.scale),Float64.(bundle.feature_weights),
                         zeros(length(FNAMES)),zeros(length(FNAMES)))
end

function fusion_reset!(st::FusionTracker)
    reset!(st.tr); reset!(st.hr)
    fill!(st.state,0.0)
    st.initialized=false
    st.ema15=0.0; st.ema60=0.0
end

# At time t, a separate forward filter is updated for each frozen K-only HMM.
# No QC label is examined. The resulting quantities are NOT K probabilities.
function fusion_event!(out::AbstractVector{Float32},st::FusionTracker)
    fill!(out,0f0)
    st.tr.count==61 || return
    oos_features!(st.rawfeat,st.tr)  # EXACT same 8 features as model training
    @inbounds for d in eachindex(st.z)
        st.z[d]=clamp((st.rawfeat[d]-st.center[d])/max(st.scale[d],1e-8)*st.fw[d],-6.0,6.0)
    end
    mixlog=-Inf
    @inbounds for c in 1:3
        m=st.models[c]
        old1=st.initialized ? st.state[1,c] : m.pi[1]
        old2=st.initialized ? st.state[2,c] : m.pi[2]
        prior1=st.initialized ? (old1*m.A[1,1]+old2*m.A[2,1]) : m.pi[1]
        prior2=st.initialized ? (old1*m.A[1,2]+old2*m.A[2,2]) : m.pi[2]
        e1=0.0; e2=0.0
        for d in eachindex(st.z)
            v1=max(m.var[d,1],1e-4); v2=max(m.var[d,2],1e-4)
            q1=st.z[d]-m.mu[d,1]; q2=st.z[d]-m.mu[d,2]
            e1+=log(2pi*v1)+q1*q1/v1
            e2+=log(2pi*v2)+q2*q2/v2
        end
        a=log(max(prior1,1e-12))-0.5*e1
        b=log(max(prior2,1e-12))-0.5*e2
        ell=oos_logadd(a,b)
        p2=exp(b-ell)
        st.state[1,c]=exp(a-ell); st.state[2,c]=p2
        out[c]=Float32(clamp(ell/10,-8,8))       # predictive log likelihood
        out[3+c]=Float32(2*p2-1)                   # filtered event-state evidence
        out[6+c]=Float32(st.initialized ? abs(p2-old2) : 0) # observed state change
        mixlog=oos_logadd(mixlog,log(max(st.weights[c],1e-12))+ell)
    end
    st.initialized=true
    surprise=clamp(-mixlog,-50,80)
    a15=exp(-1/15); a60=exp(-1/60)
    st.ema15=a15*st.ema15+(1-a15)*surprise
    st.ema60=a60*st.ema60+(1-a60)*surprise
    out[10]=Float32(clamp(surprise/10,-8,8))
    out[11]=Float32(clamp(st.ema15/10,-8,8))
    out[12]=Float32(clamp(st.ema60/10,-8,8))
    out[13]=1f0
end

# 15 weather/sensor channels + 13 event-HMM channels, all causal.
# Conventions for unavailable derivatives are zero plus their validity masks.
function fusion_minute!(out::AbstractVector{Float32},st::FusionTracker,
                        t::Float32,rh::Float32)
    fill!(out,0f0)
    tv=isfinite(t); rv=isfinite(rh)
    if tv
        out[1]=clamp((t-20f0)/15f0,-8f0,8f0)
        out[3]=1f0
    end
    if rv
        out[2]=clamp((rh-70f0)/30f0,-8f0,8f0)
        out[4]=1f0
    end
    # Physically extreme temperatures are retained for the CNN, but the HMM
    # requires continuous, plausible measurements to form temporal features.
    if !(tv && -90f0<t<65f0)
        reset!(st.tr); reset!(st.hr)
        fill!(st.state,0.0)
        st.initialized=false; st.ema15=0.0; st.ema60=0.0
        return
    end
    put!(st.tr,Float64(t))
    if rv
        put!(st.hr,Float64(rh))
    else
        reset!(st.hr)
    end
    if st.tr.count>=61
        oos_features!(st.rawfeat,st.tr)
        out[5]=Float32(clamp(st.rawfeat[1]/0.5,-8,8))  # dT1
        out[6]=Float32(clamp(st.rawfeat[2]/1.5,-8,8))  # dT5
        out[7]=Float32(clamp(st.rawfeat[3]/3,-8,8))    # dT30
        out[12]=Float32(clamp(st.rawfeat[5]/1.0,0,8)) # std T30
        out[13]=Float32(clamp(st.rawfeat[8]/2,-8,8))  # T - preceding hour
        out[14]=Float32(st.rawfeat[6])                # 15min flatness
        out[15]=Float32(clamp(st.rawfeat[7]/0.1,-8,8))# signed slope
    end
    if rv && st.hr.count>=6 && st.tr.count>=6
        rt=Float64(rh)
        dr=rt-lag(st.hr,5)
        out[8]=Float32(clamp(dr/10,-8,8))
        dt5=Float64(t)-lag(st.tr,5)
        td=fusion_dewpoint(Float64(t),rt)
        prevtd=fusion_dewpoint(lag(st.tr,5),lag(st.hr,5))
        if isfinite(td) && isfinite(prevtd)
            out[9]=Float32(clamp((Float64(t)-td)/20,0,8))
            out[10]=Float32(clamp((td-prevtd)/3,-8,8))
            # d(log water-vapor pressure) proxy: roughly zero if RH loss
            # is explained entirely by increasing temperature.
            resid=log(rt/lag(st.hr,5)) + 17.625*243.04/(243.04+Float64(t))^2*dt5
            out[11]=Float32(clamp(resid*10,-8,8))
        end
    end
    fusion_event!(@view(out[FUSION_RAW+1:end]),st)
end

struct FusionShard
    ship::String
    day::Date
    split::Symbol
    x::Matrix{Float32}  # (359+1440) x 28
    y::Vector{Int8}     # 1440 target labels, -1=ignore
    snapshot::NamedTuple # stream state *after* this day; makes caching causal
end
struct FusionRef
    path::String
    ship::String
    day::Date
    split::Symbol
    k::Int
    z::Int
end

# An HMM's filtered state cannot be reconstructed from the last 61 readings;
# preserve it across cached days so a resumed preparation is identical.
function fusion_snapshot(st::FusionTracker)
    return (t_v=copy(st.tr.v),t_next=st.tr.next,t_count=st.tr.count,
            rh_v=copy(st.hr.v),rh_next=st.hr.next,rh_count=st.hr.count,
            state=copy(st.state),initialized=st.initialized,
            ema15=st.ema15,ema60=st.ema60)
end
function fusion_restore!(st::FusionTracker,s)
    st.tr.v .= s.t_v; st.tr.next=s.t_next; st.tr.count=s.t_count
    st.hr.v .= s.rh_v; st.hr.next=s.rh_next; st.hr.count=s.rh_count
    st.state .= s.state; st.initialized=s.initialized
    st.ema15=s.ema15; st.ema60=s.ema60
end

function fusion_atomic(path,obj)
    mkpath(dirname(path))
    tmp=path*".partial"
    open(tmp,"w") do io
        serialize(io,obj)
    end
    mv(tmp,path;force=true)
end

function fusion_prepare(o,ships,bundle; include_test::Bool=false)
    ex=exclusions(o["exclude-csv"])
    mh=bytes2hex(sha256(read(o["hmm"])))
    eh=lowercase(o["exclude-csv"])=="none" ? "none" : bytes2hex(sha256(read(o["exclude-csv"])))
    key=bytes2hex(sha256(string(FUSION_VERSION,FUSION_HIST,abspath(o["root"]),mh,eh,
                              join(ships,","),o["max-days"])))[1:14]
    root=joinpath(o["cache-dir"],"fusion_"*key)
    # Existing 360-minute shards already contain the same 1,440 feature rows
    # per day. Upgrade them to 769-minute context without rescanning netCDF
    # or recalculating frozen HMM states; all features themselves are unchanged.
    oldkey=bytes2hex(sha256(string("samos_event_cnn_causal_v1",abspath(o["root"]),mh,eh,
                                 join(ships,","),o["max-days"])))[1:14]
    oldroot=joinpath(o["cache-dir"],"fusion_"*oldkey)
    mkpath(root)
    println("Prepared data cache: $(abspath(root))")
    isdir(oldroot) && println("Found older 360-minute cache; eligible days will be upgraded automatically")
    refs=Dict(s=>FusionRef[] for s in (:train,:val,:test))
    failures=0; built=0; reused=0; upgraded=0
    maxdays=parse(Int,o["max-days"])
    for ship in ships
        daysmap=fusion_files(o["root"],ship)
        days=sort(collect(keys(daysmap)))
        println("$ship: $(length(days)) eligible-year candidate daily files")
        st=FusionTracker(bundle)
        prev=Date(1,1,1); prevsplit=:ignore
        # For consecutive days, this contains the last 768 minutes of the
        # previous day; missing days, exclusions, and split boundaries reset it
        # to zeros. Measurement availability masks distinguish gaps from 0°C.
        hist=zeros(Float32,FUSION_HIST-1,FUSION_CHANNELS)
        selected=Dict(s=>0 for s in (:train,:val,:test))
        for day in days
            split=fusion_split(ship,year(day))
            split==:ignore && continue
            split==:test && !include_test && continue
            if excluded(ex,ship,day)
                fusion_reset!(st); fill!(hist,0f0)
                prev=Date(1,1,1); prevsplit=:ignore
                continue
            end
            maxdays>0 && selected[split]>=maxdays && continue
            if day!=prev+Day(1) || split!=prevsplit
                fusion_reset!(st); fill!(hist,0f0)
            end
            cache=joinpath(root,ship,"$(ship)_$(Dates.format(day,dateformat"yyyymmdd")).jls")
            if isfile(cache) && !fusion_yes(o,"rebuild-cache")
                sh=deserialize(cache)
                sh isa FusionShard || error("Wrong shard in $cache")
                size(sh.x)==(FUSION_HIST-1+FUSION_DAY,FUSION_CHANNELS) ||
                    error("Cached history shape does not match the 769-minute model: $cache")
                (sh.day==day && sh.ship==ship && sh.split==split) ||
                    error("Cache identity mismatch: $cache")
                fusion_restore!(st,sh.snapshot)
                hist=copy(@view(sh.x[end-FUSION_HIST+2:end,:]))
                k=count(==(Int8(1)),sh.y); z=count(==(Int8(0)),sh.y)
                push!(refs[split],FusionRef(cache,ship,day,split,k,z))
                selected[split]+=1; reused+=1
                prev=day; prevsplit=split
                continue
            end
            oldcache=joinpath(oldroot,ship,"$(ship)_$(Dates.format(day,dateformat"yyyymmdd")).jls")
            if !fusion_yes(o,"rebuild-cache") && isfile(oldcache)
                oldsh=deserialize(oldcache)
                oldsh isa FusionShard || error("Legacy shard type mismatch: $oldcache")
                (oldsh.day==day && oldsh.ship==ship && oldsh.split==split) ||
                    error("Legacy cache identity mismatch: $oldcache")
                size(oldsh.x)==(359+FUSION_DAY,FUSION_CHANNELS) ||
                    error("Legacy cache shape mismatch: $oldcache")
                # The old day's *measurement-derived* features are the last
                # 1440 rows. Append today's 1440 rows to the trailing 768
                # minutes from the prior contiguous day (zeros otherwise).
                full=vcat(hist,@view(oldsh.x[end-FUSION_DAY+1:end,:]))
                sh=FusionShard(ship,day,split,full,oldsh.y,oldsh.snapshot)
                fusion_atomic(cache,sh)
                fusion_restore!(st,sh.snapshot)
                hist=copy(@view(full[end-FUSION_HIST+2:end,:]))
                k=count(==(Int8(1)),sh.y); z=count(==(Int8(0)),sh.y)
                push!(refs[split],FusionRef(cache,ship,day,split,k,z))
                selected[split]+=1; upgraded+=1
                prev=day; prevsplit=split
                upgraded%500==0 && println("  Upgraded $upgraded old daily cache shards")
                continue
            end
            src=try
                fusion_read_day(daysmap[day],day)
            catch err
                failures+=1
                failures<=8 && @warn "Skipping unreadable SAMOS day" ship day reason=sprint(showerror,err)
                fusion_reset!(st); fill!(hist,0f0)
                prev=Date(1,1,1); prevsplit=:ignore
                continue
            end
            xday=zeros(Float32,FUSION_DAY,FUSION_CHANNELS)
            for minute in 1:FUSION_DAY
                fusion_minute!(@view(xday[minute,:]),st,src.t[minute],src.rh[minute])
            end
            full=vcat(hist,xday)
            sh=FusionShard(ship,day,split,full,src.y,fusion_snapshot(st))
            # Rewrites one day at a time and caches all features, including
            # previously available HMM states, for reproducibility.
            fusion_atomic(cache,sh)
            built+=1
            selected[split]+=1
            k=count(==(Int8(1)),src.y); z=count(==(Int8(0)),src.y)
            push!(refs[split],FusionRef(cache,ship,day,split,k,z))
            hist=copy(@view(full[end-FUSION_HIST+2:end,:]))
            prev=day; prevsplit=split
            if built%200==0
                println("  Prepared $built days; train=$(length(refs[:train])), val=$(length(refs[:val])), test=$(length(refs[:test]))")
            end
        end
        println("$ship prepared: ",selected)
    end
    println("Skipped $failures unreadable days; built $built from netCDF, upgraded $upgraded old shards, reused $reused new shards.")
    for s in (:train,:val,:test)

        @printf("%-5s days=%5d K=%9d Z=%9d\n",
        string(s),
        length(refs[s]),
        sum((r.k for r in refs[s]); init=0),
        sum((r.z for r in refs[s]); init=0))

    end
    return refs
end

struct FusionBlock
    first::Conv
    second::Conv
    drop::Dropout
end
Flux.@layer FusionBlock
# Nonzero negative slope avoids the hard zero-gradient side of ordinary ReLU.
# Keep the FINAL Conv linear: fusion_bce() consumes logits, and only
# fusion_probabilities() applies a sigmoid when scoring.
@inline fusion_leaky(x) = Flux.leakyrelu(x, 0.01f0)

function FusionBlock(width::Int,d::Int,rate::Float32)
    FusionBlock(Conv((3,),width=>width,fusion_leaky;pad=(2*d,0),dilation=d),
                Conv((3,),width=>width;pad=(2*d,0),dilation=d),Dropout(rate))
end
(b::FusionBlock)(x)=fusion_leaky.(x .+ b.drop(b.second(b.first(x))))

struct FusionCNN
    temporal::Chain
end
Flux.@layer FusionCNN
function FusionCNN(width::Int, rate::Float32)
    layers = [FusionBlock(width,d,rate) for d in FUSION_DILATIONS]
    return FusionCNN(Chain(
        Conv((5,),FUSION_CHANNELS=>width,fusion_leaky;pad=(4,0)),
        layers...,
        Conv((1,),width=>width,fusion_leaky),
        Conv((1,),width=>1)  # linear logits, NOT sigmoid
    ))
end
function (m::FusionCNN)(x)
    out=m.temporal(reshape(x,size(x,1),size(x,2),1))
    return vec(out)[FUSION_HIST:(FUSION_HIST+FUSION_DAY-1)]
end

function fusion_input(sh::FusionShard,mode::String)
    if mode=="raw"
        x=copy(sh.x)
        @views x[:,FUSION_RAW+1:end].=0f0
        return x
    end
    return sh.x
end

function fusion_bce(logits,y,pos_weight)
    active=Float32.(y .>= 0)
    target=Float32.(y .== 1)
    weight=1f0 .+ (pos_weight-1f0).*target
    # Stable BCE with logits: max(z,0)-z*y+log(1+exp(-abs(z))).
    bce=max.(logits,0f0).-logits.*target.+log1p.(exp.(-abs.(logits)))
    return sum(active.*weight.*bce)/max(sum(active.*weight),1f0)
end

fusion_load(r::FusionRef)=deserialize(r.path)

function fusion_probabilities(model,sh,mode)
    return Flux.sigmoid.(model(fusion_input(sh,mode)))
end

function fusion_validate(model,refs,mode; predicted_io=nothing)
    Flux.testmode!(model)
    positive=Float32[]; negative=Float32[]
    # Accumulate all labeled scores: no sampled validation or test points.
    for (j,r) in enumerate(refs)
        sh=fusion_load(r)
        p=fusion_probabilities(model,sh,mode)
        @inbounds for i in eachindex(sh.y)
            yy=sh.y[i]
            if yy==1
                push!(positive,p[i])
            elseif yy==0
                push!(negative,p[i])
            end
            if predicted_io!==nothing && yy>=0
                stamp=DateTime(sh.day)+Minute(i-1)
                println(predicted_io,"$(sh.ship),$stamp,$(Int(yy)),$(p[i])")
            end
        end
        j%200==0 && println("  Scored $j / $(length(refs)) days")
    end
    return positive,negative
end

function fusion_threshold(neg,target)
    isempty(neg) && error("No valid Z labels in validation")
    sort!(neg)
    allowed=floor(Int,target*length(neg))
    return neg[max(1,length(neg)-allowed)]
end
function fusion_metrics(pos,neg,thr)
    tp=count(>(thr),pos); fp=count(>(thr),neg)
    nk=length(pos); nz=length(neg)
    return (K=nk,Z=nz,TP=tp,FP=fp,
            recall=nk==0 ? NaN : tp/nk,
            fpr=nz==0 ? NaN : fp/nz,
            precision=tp+fp==0 ? 0.0 : tp/(tp+fp))
end

# Exact AUROC and average precision, grouped across equal scores.
# The diagnostic uses EVERY labeled minute, not a sampled subset.
function fusion_auc_ap(pos,neg)
    P=length(pos); N=length(neg)
    P>0 && N>0 || return (roc=NaN,ap=NaN)
    sort!(pos;rev=true); sort!(neg;rev=true)
    ip=1; iz=1; tp=0; fp=0
    prevtp=0.0; prevfp=0.0; roc=0.0; ap=0.0
    while ip<=P || iz<=N
        a=ip<=P ? pos[ip] : -Inf32
        b=iz<=N ? neg[iz] : -Inf32
        score=max(a,b)
        while ip<=P && pos[ip]==score
            tp+=1; ip+=1
        end
        while iz<=N && neg[iz]==score
            fp+=1; iz+=1
        end
        tpr=tp/P; fpr=fp/N
        roc+=(fpr-prevfp)*(tpr+prevtp)/2
        ap+=(tpr-prevtp)*tp/(tp+fp)
        prevtp=tpr; prevfp=fpr
    end
    return (roc=roc,ap=ap)
end

# Event-level recall: a true K episode (>=5 consecutive K minutes) is detected
# if ANY of its flagged minutes exceeds the frozen threshold.
# Day boundaries are deliberately independent in this diagnostic; otherwise
# whole-voyage event segmentation must be implemented separately.
function fusion_event_metrics(model,refs,mode,thr)
    observed=0; caught=0; alert_runs=0; false_runs=0; totaldays=0
    for r in refs
        sh=fusion_load(r)
        p=fusion_probabilities(model,sh,mode)
        totaldays+=1
        run=0; hit=false
        active=false; contaminated=false
        function close_k!()
            if run>=5
                observed+=1
                hit && (caught+=1)
            end
            run=0; hit=false
        end
        function close_alert!()
            if active
                alert_runs+=1
                !contaminated && (false_runs+=1)
            end
            active=false; contaminated=false
        end
        for i in 1:FUSION_DAY
            yy=sh.y[i]
            if yy==1
                run+=1
                hit |= p[i]>thr
            else
                close_k!()
            end
            pred=(yy>=0 && p[i]>thr)
            if pred
                active=true
                contaminated |= yy==1
            else
                close_alert!()
            end
        end
        close_k!(); close_alert!()
    end
    return (K_events=observed,detected_events=caught,
            event_recall=observed==0 ? NaN : caught/observed,
            alerts=alert_runs,Z_only_alerts=false_runs,
            Z_only_alerts_per_day=false_runs/max(totaldays,1))
end

function fusion_train!(o,refs)
    nk=sum(r.k for r in refs[:train]); nz=sum(r.z for r in refs[:train])
    nk>0 && nz>0 || error("Not enough training K/Z labels")
    !isempty(refs[:val]) || error("No validation days")
    pw=Float32((nz/nk)^parse(Float64,o["pos-weight-power"]))
    mode=o["mode"]
    println("Training CNN $mode | K=$nk Z=$nz | positive weight=$pw")
    Random.seed!(parse(Int,o["seed"]))
    net=FusionCNN(parse(Int,o["width"]),parse(Float32,o["dropout"]))
    println("Trainable parameters: ",sum(length,Flux.trainables(net)))
    opt=Flux.setup(Flux.Adam(parse(Float32,o["lr"])),net)
    best=nothing; best_recall=-Inf; best_thr=NaN32
    patience=parse(Int,o["patience"]); stale=0
    root=joinpath(o["output"],mode)
    mkpath(root)
    rng=MersenneTwister(parse(Int,o["seed"]))
    for ep in 1:parse(Int,o["epochs"])
        Flux.trainmode!(net)
        inds=randperm(rng,length(refs[:train]))
        totals=0.0
        for (j,idx) in enumerate(inds)
            r=refs[:train][idx]
            sh=fusion_load(r)
            x=fusion_input(sh,mode)
            # Weight each day's gradient by its fraction of eligible K/Z minutes.
            dayweight=Float32((r.k+r.z)/FUSION_DAY)
            dayweight==0f0 && continue
            loss,grad=Flux.withgradient(net) do m
                dayweight*fusion_bce(m(x),sh.y,pw)
            end
            isfinite(loss) || error("Nonfinite CNN loss on $(r.ship) $(r.day)")
            Flux.update!(opt,net,grad[1])
            totals+=Float64(loss)
            j%250==0 && @printf("Epoch %d trained %d/%d days, mean day loss %.5f\n",
                                ep,j,length(inds),totals/j)
        end
        pos,neg=fusion_validate(net,refs[:val],mode)
        thr=fusion_threshold(neg,parse(Float64,o["target-fpr"]))
        m=fusion_metrics(pos,neg,thr)
        ra=fusion_auc_ap(pos,neg)
        @printf("Epoch %d | loss=%.5f | VAL recall=%.4f FPR=%.4f precision=%.4f AUC=%.4f AP=%.4f | threshold=%.7f\n",
                ep,totals/length(inds),m.recall,m.fpr,m.precision,ra.roc,ra.ap,thr)
        if isfinite(m.recall) && m.recall>best_recall+1e-5
            best=deepcopy(net); best_recall=m.recall; best_thr=thr; stale=0
            fusion_atomic(joinpath(root,"best_checkpoint.jls"),
                          (model=best,threshold=thr,epoch=ep,
                           mode=mode,config=o,hmm=o["hmm"],
                            version=FUSION_VERSION,history=FUSION_HIST))
        else
            stale+=1
        end
        stale>=patience && (println("Early stopping after $ep epochs"); break)
    end
    best===nothing && error("Could not select a model on validation")
    Flux.testmode!(best)
    pos,neg=fusion_validate(best,refs[:val],mode)
    best_thr=fusion_threshold(neg,parse(Float64,o["target-fpr"]))
    println("FROZEN VALIDATION: ",fusion_metrics(pos,neg,best_thr), " ",fusion_auc_ap(pos,neg))
    println("VALIDATION EVENTS: ",fusion_event_metrics(best,refs[:val],mode,best_thr))
    return best,best_thr
end

function fusion_test!(o,refs,net,thr)
    isempty(refs[:test]) && error("No test days")
    mode=o["mode"]
    println("UNTOUCHED TEST: fixed threshold from validation = $thr")
    dest=joinpath(o["output"],mode)
    predpath=joinpath(dest,"test_predictions.csv")
    if fusion_yes(o,"write-predictions")
        open(predpath,"w") do io
            println(io,"ship,time_utc,y_K1_Z0,p_K")
            pos,neg=fusion_validate(net,refs[:test],mode;predicted_io=io)
            println("TEST MINUTES: ",fusion_metrics(pos,neg,thr), " ",fusion_auc_ap(pos,neg))
        end
    else
        pos,neg=fusion_validate(net,refs[:test],mode)
        println("TEST MINUTES: ",fusion_metrics(pos,neg,thr), " ",fusion_auc_ap(pos,neg))
    end
    println("TEST EVENTS: ",fusion_event_metrics(net,refs[:test],mode,thr))
    for ship in unique(r.ship for r in refs[:test])
        sub=filter(r->r.ship==ship,refs[:test])
        p,z=fusion_validate(net,sub,mode)
        println("  $ship TEST MINUTES: ",fusion_metrics(p,z,thr), " ",fusion_auc_ap(p,z))
    end
end

function fusion_selftest()
    # Model's event channels change with the input signal, and all channels are finite.
    dummy=GHMM([0.5,0.5],[0.96 0.04; 0.04 0.96],
               zeros(length(FNAMES),2),ones(length(FNAMES),2))
    bun=(models=[deepcopy(dummy) for _ in 1:3],weights=fill(1/3,3),
         center=zeros(8),scale=ones(8),feature_weights=ones(8),feature_names=FNAMES)
    tr=FusionTracker(bun)
    xx=zeros(Float32,120,FUSION_CHANNELS)
    for i in 1:120
        fusion_minute!(@view(xx[i,:]),tr,Float32(24+0.06*sin(i/8)+0.007*i),Float32(78-0.3*cos(i/9)))
    end
    all(isfinite,xx) || error("Nonfinite event inputs")
    xx[120,end]==1 || error("Event tracker never warmed up")
    FUSION_RF==769 || error("Incorrect CNN receptive field")
    net=FusionCNN(16,0.0f0)
    y=rand(Int8[-1,0,1],FUSION_DAY)
    input=randn(Float32,FUSION_HIST-1+FUSION_DAY,FUSION_CHANNELS)
    logits=net(input)
    length(logits)==FUSION_DAY || error("Bad CNN output length")
    all(isfinite,logits) || error("Non-finite CNN logits")
    loss,grads=Flux.withgradient(net) do m
        fusion_bce(m(input),y,5f0)
    end
    opt=Flux.setup(Flux.Adam(1f-3),net)
    Flux.update!(opt,net,grads[1])
    isfinite(loss) || error("Bad test loss")
    # A causal model's earlier output cannot depend on future input.
    Flux.testmode!(net)
    a=net(input)
    altered=copy(input)
    altered[end-99:end,:].+=100f0
    b=net(altered)
    maximum(abs.(a[1:end-100].-b[1:end-100]))<1e-4 ||
        error("CNN future data leakage: changed earlier predictions")
    println("PASS: RF=$(FUSION_RF) min, 28 finite channels, frozen HMM filtering, CNN forward/backward, causality")
end

function fusion_main()
    o,ships=fusion_options()
    if fusion_yes(o,"self-test")
        fusion_selftest(); return
    end
    isdir(o["root"]) || error("Missing selected-data directory $(o["root"])")
    isfile(o["hmm"]) || error("No frozen K-only HMM: $(o["hmm"])\nTrain it with 04_k_shape_mixture.jl first.")
    bun=deserialize(o["hmm"])
    bun.training_only || error("HMM checkpoint must be trained on TRAIN-ONLY K episodes")
    println("FROZEN K-only event encoder (3 x 2-state HMMs) + causal CNN")
    println("CNN receptive field: $(FUSION_RF) minutes; prior-day input: $(FUSION_HIST-1) minutes")
    println("Hidden activation: LeakyReLU(0.01), linear logit output")
    println("Raw: T/RH and thermodynamic behavior; event: 3x likelihood, state and shift, short/long surprise")
    println("Mode=$(o["mode"]), vessels=$(join(ships,",")); 2025 remains untouched until testing.")
    @warn "Exclusion date CSV was marked provisional. Verify with Kristen before final scientific reporting."
    # Only load 2025 / ZCYL5-2021 when the user explicitly runs the frozen test.
    refs=fusion_prepare(o,ships,bun;
                        include_test=(fusion_yes(o,"test-only") || fusion_yes(o,"test")))
    if fusion_yes(o,"prepare-only")
        println("Prepared chronological daily cache only. No CNN training.")
        return
    end
    if fusion_yes(o,"test-only")
        path=joinpath(o["output"],o["mode"],"best_checkpoint.jls")
        isfile(path) || error("No saved best checkpoint: $path")
        saved=deserialize(path)
        saved.mode==o["mode"] || error("Checkpoint mode mismatch")
        hasproperty(saved,:version) && saved.version==FUSION_VERSION ||
            error("Checkpoint architecture version does not match this model")
        hasproperty(saved,:history) && saved.history==FUSION_HIST ||
            error("Checkpoint history length does not match this model")
        saved.hmm==o["hmm"] || error("Checkpoint HMM path mismatch")
        for key in ("ships","root","exclude-csv","max-days","width")
            saved.config[key]==o[key] || error("Checkpoint data/config mismatch at --$key")
        end
        Flux.testmode!(saved.model)
        fusion_test!(o,refs,saved.model,saved.threshold)
        return
    end
    net,thr=fusion_train!(o,refs)
    fusion_atomic(joinpath(o["output"],o["mode"],"final_model.jls"),
                  (model=net,threshold=thr,config=o,hmm=o["hmm"],
                    version=FUSION_VERSION,history=FUSION_HIST))
    fusion_yes(o,"test") && fusion_test!(o,refs,net,thr)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    fusion_main()
end