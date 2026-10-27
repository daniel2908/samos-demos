using NCDatasets
using Flux
using Statistics
using Random
using Dates
using Printf

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
const ROOT = get(ENV, "SAMOS_ROOT", ".")
const BOATS = ["WTEO", "WTER", "WTEE", "WTEA"]

const TRAIN_END = 2023
const VAL_YEAR = 2024
const TEST_YEAR = 2025

const WINDOWS = (15, 30, 60, 120)
const WINDOW = maximum(WINDOWS)
const TRAIN_MAX_K = 150_000
const TRAIN_MAX_Z = 300_000
const VAL_MAX_K = 50_000
const VAL_MAX_Z = 150_000

const FOCAL_ALPHA = 0.75f0
const FOCAL_GAMMA = 2f0
const LEARNING_RATE = 1f-2
const BATCH_SIZE = 4096
const EVAL_BATCH_SIZE = 8192
const MAX_EPOCHS = 1000
const PATIENCE = 50
const MIN_DELTA = 1f-4
const SEED = 42

Random.seed!(SEED)

# -----------------------------------------------------------------------------
# Read SAMOS temperature + K/Z flags
# -----------------------------------------------------------------------------
function file_date(path)
    m = match(r"_(\d{4})(\d{2})(\d{2})v", basename(path))
    m === nothing && return nothing
    Date(parse(Int, m.captures[1]), parse(Int, m.captures[2]), parse(Int, m.captures[3]))
end

function vessel_files(boat)
    dir = joinpath(ROOT, "$(boat)-research-q")
    isdir(dir) || return String[]
    sort(filter(f -> endswith(lowercase(f), ".nc"), readdir(dir; join=true)))
end

aschar(x) = ismissing(x) ? '?' : x isa Char ? x : x isa Integer ? Char(x) : first(string(x))

function temperature_flags(ds)
    tvar = ds["T"]
    haskey(tvar.attrib, "qcindex") || error("T has no qcindex")
    qi = Int(tvar.attrib["qcindex"])

    flagvar = ds["flag"]
    raw = Array(flagvar)
    dims = String.(NCDatasets.dimnames(flagvar))
    time_dim = findfirst(==("time"), dims)
    time_dim === nothing && error("No time dimension in flag variable")

    flags = time_dim == 1 ? raw[:, qi] : raw[qi, :]
    aschar.(vec(flags))
end

function read_day(path)
    NCDataset(path, "r") do ds
        (!haskey(ds, "T") || !haskey(ds, "flag")) && return nothing

        rawT = vec(Array(ds["T"]))
        flags = temperature_flags(ds)
        n = min(length(rawT), length(flags))
        T = fill(NaN, n)

        for i in 1:n
            x = rawT[i]
            ismissing(x) && continue
            v = Float64(x)
            if isfinite(v) && -100 < v < 70
                T[i] = v
            end
        end

        return T, flags[1:n]
    end
end

# -----------------------------------------------------------------------------
# Four rolling line-length features
# -----------------------------------------------------------------------------
function rolling_line_length(T, window)
    n = length(T)
    result = fill(Float32(NaN), n)
    delta = fill(NaN, n)

    for i in 2:n
        if isfinite(T[i]) && isfinite(T[i - 1])
            delta[i] = abs(T[i] - T[i - 1])
        end
    end

    prefix = zeros(Float64, n + 1)
    bad = zeros(Int, n + 1)
    for i in 1:n
        prefix[i + 1] = prefix[i]
        bad[i + 1] = bad[i]
        if isfinite(delta[i])
            prefix[i + 1] += delta[i]
        else
            bad[i + 1] += 1
        end
    end

    for i in window:n
        firstdiff = i - window + 2
        if bad[i + 1] == bad[firstdiff]
            result[i] = Float32((prefix[i + 1] - prefix[firstdiff]) / (window - 1))
        end
    end

    result
end

function line_length_features(T)
    F = Matrix{Float32}(undef, length(WINDOWS), length(T))
    for (j, w) in enumerate(WINDOWS)
        F[j, :] .= rolling_line_length(T, w)
    end
    F
end

function eligible_indices(F, flags)
    n = min(size(F, 2), length(flags))
    idx = Int[]
    for i in WINDOW:n
        (flags[i] == 'K' || flags[i] == 'Z') || continue
        all(isfinite, @view F[:, i]) && push!(idx, i)
    end
    idx
end

function reservoir_slot(seen, stored, limit, rng)
    seen += 1
    if stored < limit
        stored += 1
        return seen, stored, stored
    end
    j = rand(rng, 1:seen)
    return seen, stored, (j <= limit ? j : 0)
end

function collect_training(boats, year_filter; max_k, max_z, seed)
    rng = MersenneTwister(seed)
    nf = length(WINDOWS)
    Xk = Matrix{Float32}(undef, nf, max_k)
    Xz = Matrix{Float32}(undef, nf, max_z)

    seen_k = stored_k = 0
    seen_z = stored_z = 0

    for boat in boats, path in vessel_files(boat)
        day = file_date(path)
        day === nothing && continue
        year_filter(year(day)) || continue

        data = try
            read_day(path)
        catch err
            @warn "Skipping unreadable file" path exception=err
            continue
        end
        data === nothing && continue

        T, flags = data
        F = line_length_features(T)

        for i in eligible_indices(F, flags)
            if flags[i] == 'K'
                seen_k, stored_k, slot = reservoir_slot(seen_k, stored_k, max_k, rng)
                slot == 0 && continue
                Xk[:, slot] .= @view F[:, i]
            else
                seen_z, stored_z, slot = reservoir_slot(seen_z, stored_z, max_z, rng)
                slot == 0 && continue
                Xz[:, slot] .= @view F[:, i]
            end
        end
    end

    n = stored_k + stored_z
    X = Matrix{Float32}(undef, nf, n)
    y = Matrix{Float32}(undef, 1, n)

    X[:, 1:stored_k] .= @view Xk[:, 1:stored_k]
    y[:, 1:stored_k] .= 1f0

    if stored_z > 0
        r = stored_k+1:n
        X[:, r] .= @view Xz[:, 1:stored_z]
        y[:, r] .= 0f0
    end

    order = randperm(rng, n)
    println("  K retained: $stored_k / $seen_k")
    println("  Z retained: $stored_z / $seen_z")
    return X[:, order], y[:, order]
end

function fit_scaler(X)
    mu = mean(X; dims=2)
    sigma = max.(std(X; dims=2, corrected=false), 1f-6)
    return mu, sigma
end

standardize!(X, mu, sigma) = (X .= (X .- mu) ./ sigma)

# -----------------------------------------------------------------------------
# Small MLP + focal loss
# -----------------------------------------------------------------------------
function make_model()
    Chain(
        Dense(length(WINDOWS) => 32, relu),
        Dense(32 => 16, relu),
        Dense(16 => 8, relu),
        Dense(8 => 1),
    )
end

function focal_loss(logits, y)
    s = (2f0 .* y .- 1f0) .* logits
    log_loss = Flux.softplus.(-s)
    one_minus_pt = exp.(-Flux.softplus.(s))
    alpha = ifelse.(y .> 0.5f0, FOCAL_ALPHA, 1f0 - FOCAL_ALPHA)
    mean(alpha .* one_minus_pt.^FOCAL_GAMMA .* log_loss)
end

function validation_loss(model, X, y)
    total = 0.0
    n = size(X, 2)
    for start in 1:EVAL_BATCH_SIZE:n
        idx = start:min(start + EVAL_BATCH_SIZE - 1, n)
        total += Float64(focal_loss(model(@view X[:, idx]), @view(y[:, idx]))) * length(idx)
    end
    total / n
end

function train_model(model, X, y, Xval, yval)
    opt = Flux.setup(Adam(LEARNING_RATE), model)
    rng = MersenneTwister(SEED)
    indices = collect(1:size(X, 2))

    best_model = deepcopy(model)
    best_loss = Inf
    stale = 0

    for epoch in 1:MAX_EPOCHS
        shuffle!(rng, indices)
        train_total = 0.0

        for start in 1:BATCH_SIZE:length(indices)
            idx = indices[start:min(start + BATCH_SIZE - 1, length(indices))]
            xb = @view X[:, idx]
            yb = @view y[:, idx]

            loss, grads = Flux.withgradient(model) do m
                focal_loss(m(xb), yb)
            end
            Flux.update!(opt, model, grads[1])
            train_total += Float64(loss) * length(idx)
        end

        train_loss = train_total / size(X, 2)
        val_loss = validation_loss(model, Xval, yval)
        @printf("epoch %4d  train %.6f  val %.6f\n", epoch, train_loss, val_loss)

        if val_loss < best_loss - MIN_DELTA
            best_loss = val_loss
            best_model = deepcopy(model)
            stale = 0
        else
            stale += 1
            stale >= PATIENCE && break
        end
    end

    best_model
end

function predict_scores(model, X)
    scores = Vector{Float32}(undef, size(X, 2))
    for start in 1:EVAL_BATCH_SIZE:length(scores)
        idx = start:min(start + EVAL_BATCH_SIZE - 1, length(scores))
        scores[idx] .= Flux.sigmoid.(vec(model(@view X[:, idx])))
    end
    scores
end

function score_year(model, mu, sigma, boats, year_filter)
    scores = Float32[]
    labels = Float32[]

    for boat in boats, path in vessel_files(boat)
        day = file_date(path)
        day === nothing && continue
        year_filter(year(day)) || continue

        data = try
            read_day(path)
        catch err
            @warn "Skipping unreadable file" path exception=err
            continue
        end
        data === nothing && continue

        T, flags = data
        F = line_length_features(T)
        idx = eligible_indices(F, flags)
        isempty(idx) && continue

        X = Matrix{Float32}(F[:, idx])
        standardize!(X, mu, sigma)

        append!(scores, predict_scores(model, X))
        append!(labels, (flags[i] == 'K' ? 1f0 : 0f0 for i in idx))
    end

    return scores, labels
end

# -----------------------------------------------------------------------------
# Threshold + metrics
# -----------------------------------------------------------------------------
function metrics(scores, labels, threshold)
    truth = labels .> 0.5f0
    pred = scores .>= threshold

    tp = sum(pred .& truth)
    tn = sum(.!pred .& .!truth)
    fp = sum(pred .& .!truth)
    fn = sum(.!pred .& truth)

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 2tp / max(2tp + fp + fn, 1)

    (; tp, tn, fp, fn, recall, precision, fpr, accuracy, f1)
end

function choose_threshold(scores, labels)
    best_t = 0.5
    best_f1 = -1.0
    for t in 0.01:0.01:0.99
        f1 = metrics(scores, labels, t).f1
        if f1 > best_f1
            best_f1 = f1
            best_t = t
        end
    end
    best_t
end

function print_metrics(title, scores, labels, threshold)
    m = metrics(scores, labels, threshold)
    println("\n$title")
    @printf("  threshold   %.2f\n", threshold)
    @printf("  K recall    %6.2f %%\n", 100m.recall)
    @printf("  K precision %6.2f %%\n", 100m.precision)
    @printf("  Z FPR       %6.2f %%\n", 100m.fpr)
    @printf("  accuracy    %6.2f %%\n", 100m.accuracy)
    @printf("  F1          %6.2f %%\n", 100m.f1)
    println("  TP/FN = $(m.tp)/$(m.fn), FP/TN = $(m.fp)/$(m.tn)")
end

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
println("\nWAVE LINE-LENGTH MLP")
println("features = ", join(WINDOWS, ", "), " samples")
println("train <= $TRAIN_END, validation = $VAL_YEAR, test = $TEST_YEAR")

println("\nCollecting training data...")
Xtrain, ytrain = collect_training(BOATS, y -> y <= TRAIN_END;
                                  max_k=TRAIN_MAX_K, max_z=TRAIN_MAX_Z, seed=SEED)
mu, sigma = fit_scaler(Xtrain)
standardize!(Xtrain, mu, sigma)

println("\nCollecting validation sample for early stopping...")
Xval, yval = collect_training(BOATS, y -> y == VAL_YEAR;
                              max_k=VAL_MAX_K, max_z=VAL_MAX_Z, seed=SEED + 1)
standardize!(Xval, mu, sigma)

println("\nTraining...")
model = train_model(make_model(), Xtrain, ytrain, Xval, yval)
Xtrain = ytrain = Xval = yval = nothing
GC.gc()

println("\nScoring full $VAL_YEAR validation year...")
val_scores, val_labels = score_year(model, mu, sigma, BOATS, y -> y == VAL_YEAR)
threshold = choose_threshold(val_scores, val_labels)
print_metrics("VALIDATION $VAL_YEAR", val_scores, val_labels, threshold)

for boat in BOATS
    println("\nScoring $boat / $TEST_YEAR...")
    scores, labels = score_year(model, mu, sigma, [boat], y -> y == TEST_YEAR)
    isempty(scores) ? println("  no usable observations") :
                      print_metrics("$boat TEST $TEST_YEAR", scores, labels, threshold)
end

println("\nDone.")
