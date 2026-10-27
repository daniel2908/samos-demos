const MODEL_TITLE = "Waveform line-length MLP with focal loss"
const OUTPUT_NAME = "line_length_simple_results"
using NCDatasets
using Flux
using Statistics
using Random
using Dates
using Printf
using Serialization

# T is air temperature; K is flagged and Z is good.
const DATA_DIRECTORY = get(ENV, "SAMOS_ROOT", ".")
const VESSELS = ["WTEO", "WTER", "WTEE", "WTEA"]
const TRAIN_END_YEAR = 2023
const VALIDATION_YEAR = 2024
const TEST_YEAR = 2025

# A prediction needs 120 valid samples from the same daily file.
const WINDOW_SIZE = 120
const FOCAL_ALPHA = 0.75f0
const FOCAL_GAMMA = 2f0
const LEARNING_RATE = 1f-2
const L2_LAMBDA = 0f0
const BATCH_SIZE = 4096
const EVAL_BATCH_SIZE = 1024
const MAX_EPOCHS = 5000
const EARLY_STOP_PATIENCE = 100
const MIN_VALIDATION_IMPROVEMENT = 1f-4

# Select the threshold on validation data, then apply it to the test year.
const THRESHOLD_MODE = :f1
const MAX_VALIDATION_FPR = 0.05
const THRESHOLD_BINS = 1000

const EARLY_STOP_SAMPLE_SIZE = 150_000
const RANDOM_SEED = 42
const OUTPUT_DIRECTORY = joinpath(DATA_DIRECTORY, OUTPUT_NAME)
const LINE_LENGTH_WINDOWS = (15, 30, 60, 120)
const INPUT_SIZE = length(LINE_LENGTH_WINDOWS)
const MAX_TRAIN_K = typemax(Int)
const MAX_TRAIN_Z = typemax(Int)

function make_model()
    return Chain(Dense(INPUT_SIZE => 32, relu),
                 Dense(32 => 16, relu),
                 Dense(16 => 8, relu),
                 Dense(8 => 1))
end

function focal_loss(logits, labels)
    signed_logits = (2f0 .* labels .- 1f0) .* logits
    log_loss = Flux.softplus.(-signed_logits)
    wrong_class_probability = exp.(-Flux.softplus.(signed_logits))
    class_weight = ifelse.(labels .> 0.5f0, FOCAL_ALPHA, 1f0 - FOCAL_ALPHA)
    return mean(class_weight .* (wrong_class_probability .^ FOCAL_GAMMA) .* log_loss)
end

function main()
    Random.seed!(RANDOM_SEED)
    mkpath(OUTPUT_DIRECTORY)
    open(joinpath(OUTPUT_DIRECTORY, "metrics.csv"), "w") do output_file
        println(output_file, "split,threshold,tp,fn,fp,tn,recall,precision,fpr,specificity,accuracy,balanced_accuracy,f1,k_prevalence,roc_auc_approx,average_precision_approx,k_eligible,k_observed,k_coverage,z_eligible,z_observed,z_coverage")
    end
    println("\n$MODEL_TITLE")
    println("Training through $TRAIN_END_YEAR; validation $VALIDATION_YEAR; test $TEST_YEAR")

    println("\nLoading training data")
    training_features, training_labels = collect_samples(
        VESSELS, file_year -> file_year <= TRAIN_END_YEAR; seed=RANDOM_SEED)
    size(training_features, 2) > 0 || error("No training data; check DATA_DIRECTORY and downloaded files")
    preprocessing = fit_feature_scaler(training_features)
    scale_features!(training_features, preprocessing)

    println("\nLoading validation sample for early stopping")
    validation_features, validation_labels = collect_samples(
        VESSELS, file_year -> file_year == VALIDATION_YEAR; uniform=true, seed=RANDOM_SEED + 1)
    size(validation_features, 2) > 0 || error("No validation data for early stopping")
    scale_features!(validation_features, preprocessing)

    model, best_epoch = train_model(make_model(), training_features, training_labels,
                                    validation_features, validation_labels)
    println("Restored model from epoch $best_epoch")
    training_features = training_labels = validation_features = validation_labels = nothing
    GC.gc()

    println("\nScoring full $VALIDATION_YEAR validation data")
    validation_data = score_holdout(model, preprocessing, VESSELS, file_year -> file_year == VALIDATION_YEAR)
    isempty(validation_data.scores) && error("No eligible validation observations")
    threshold = choose_threshold(threshold_curves(validation_data.scores, validation_data.labels))
    @printf("Validation threshold: %.4f (%s)\n", threshold, string(THRESHOLD_MODE))
    report_split(validation_data, "validation_$(VALIDATION_YEAR)", "Validation $VALIDATION_YEAR", threshold)
    validation_data = nothing
    GC.gc()

    serialize(joinpath(OUTPUT_DIRECTORY, "trained_model.jls"),
              (model=model, preprocessing=preprocessing, threshold=threshold, model_title=MODEL_TITLE,
               focal_alpha=FOCAL_ALPHA, focal_gamma=FOCAL_GAMMA, train_end=TRAIN_END_YEAR,
               validation_year=VALIDATION_YEAR, window=WINDOW_SIZE))

    for vessel in VESSELS
        println("\nScoring $vessel / $TEST_YEAR")
        test_data = score_holdout(model, preprocessing, [vessel], file_year -> file_year == TEST_YEAR)
        report_split(test_data, "$(vessel)_$(TEST_YEAR)", "$vessel test $TEST_YEAR", threshold)
        GC.gc()
    end
    println("\nMetrics and model saved to $OUTPUT_DIRECTORY")
end

function file_date(file_path::AbstractString)
    date_match = match(r"_(\d{4})(\d{2})(\d{2})v", basename(file_path))
    date_match === nothing && return nothing
    return Date(parse(Int, date_match.captures[1]), parse(Int, date_match.captures[2]),
                parse(Int, date_match.captures[3]))
end

function vessel_files(vessel::AbstractString)
    vessel_directory = joinpath(DATA_DIRECTORY, "$(vessel)-research-q")
    if !isdir(vessel_directory)
        @warn "Missing vessel directory" vessel vessel_directory
        return String[]
    end
    return sort(filter(candidate_file -> endswith(lowercase(candidate_file), ".nc"), readdir(vessel_directory; join=true)))
end

function flag_character(value)
    ismissing(value) && return '?'
    value isa Char && return value
    value isa Integer && return Char(value)
    value isa AbstractString && return isempty(value) ? '?' : first(value)
    return first(string(value))
end

function temperature_flags(dataset)
    temperature_variable = dataset["T"]
    haskey(temperature_variable.attrib, "qcindex") || error("T has no qcindex attribute")
    flag_index = Int(temperature_variable.attrib["qcindex"])
    flag_variable = dataset["flag"]
    flag_data = Array(flag_variable)
    dimension_names = String.(NCDatasets.dimnames(flag_variable))
    time_dimension = findfirst(==("time"), dimension_names)
    time_dimension === nothing && error("No time dimension in flag variable: $dimension_names")
    quality_flags = if time_dimension == 1
        flag_data[:, flag_index]
    elseif time_dimension == 2
        flag_data[flag_index, :]
    else
        error("Unexpected flag dimensions: $dimension_names")
    end
    return flag_character.(vec(quality_flags))
end

function read_day(file_path::AbstractString)
    return NCDataset(file_path, "r") do dataset
        (!haskey(dataset, "T") || !haskey(dataset, "flag")) && return nothing
        raw_temperatures = vec(Array(dataset["T"]))
        quality_flags = temperature_flags(dataset)
        sample_count = min(length(raw_temperatures), length(quality_flags))
        temperatures = fill(NaN, sample_count)
        for sample_index in 1:sample_count
            value = raw_temperatures[sample_index]
            ismissing(value) && continue
            temperature_value = Float64(value)
            if isfinite(temperature_value) && -100 < temperature_value < 70
                temperatures[sample_index] = temperature_value
            end
        end
        return temperatures, quality_flags[1:sample_count]
    end
end

function eligible_indices(temperatures, quality_flags)
    sample_count = min(length(temperatures), length(quality_flags))
    sample_count < WINDOW_SIZE && return Int[]
    invalid_prefix = zeros(Int, sample_count + 1)
    for sample_index in 1:sample_count
        invalid_prefix[sample_index + 1] = invalid_prefix[sample_index] + (!isfinite(temperatures[sample_index]))
    end
    sample_indices = Int[]
    sizehint!(sample_indices, sample_count)
    for sample_index in WINDOW_SIZE:sample_count
        (quality_flags[sample_index] == 'K' || quality_flags[sample_index] == 'Z') || continue
        first_index = sample_index - WINDOW_SIZE + 1
        invalid_prefix[sample_index + 1] == invalid_prefix[first_index] || continue
        push!(sample_indices, sample_index)
    end
    return sample_indices
end

function rolling_line_length(temperatures::Vector{Float64}, window_size::Int)
    sample_count = length(temperatures)
    line_length = fill(Float32(NaN), sample_count)
    differences = fill(NaN, sample_count)
    for sample_index in 2:sample_count
        if isfinite(temperatures[sample_index]) && isfinite(temperatures[sample_index - 1])
            differences[sample_index] = abs(temperatures[sample_index] - temperatures[sample_index - 1])
        end
    end
    difference_prefix = zeros(Float64, sample_count + 1)
    invalid_prefix = zeros(Int, sample_count + 1)
    for sample_index in 1:sample_count
        difference_prefix[sample_index + 1] = difference_prefix[sample_index]
        invalid_prefix[sample_index + 1] = invalid_prefix[sample_index]
        if isfinite(differences[sample_index])
            difference_prefix[sample_index + 1] += differences[sample_index]
        else
            invalid_prefix[sample_index + 1] += 1
        end
    end
    for sample_index in window_size:sample_count
        first_difference = sample_index - window_size + 2
        if invalid_prefix[sample_index + 1] == invalid_prefix[first_difference]
            line_length[sample_index] = Float32((difference_prefix[sample_index + 1] - difference_prefix[first_difference]) / (window_size - 1))
        end
    end
    return line_length
end

function calculate_daily_features(temperatures)
    line_length_features = Matrix{Float32}(undef, INPUT_SIZE, length(temperatures))
    for (feature_index, window) in enumerate(LINE_LENGTH_WINDOWS)
        line_length_features[feature_index, :] .= rolling_line_length(temperatures, window)
    end
    return line_length_features
end

function fill_window_features!(destination, temperatures, day_features, sample_index)
    destination .= @view day_features[:, sample_index]
end

function fit_feature_scaler(features)
    feature_means = mean(features; dims=2)
    feature_stds = max.(std(features; dims=2, corrected=false), 1f-6)
    return (mu=feature_means, sigma=feature_stds)
end

function scale_features!(features, preprocessing)
    features .= (features .- preprocessing.mu) ./ preprocessing.sigma
    return features
end

make_model_input(features, sample_indices) = features[:, sample_indices]

mutable struct Reservoir
    features::Matrix{Float32}
    labels::Vector{Float32}
    stored::Int
    seen::Int
    limit::Int
end

function Reservoir(limit::Int)
    limit > 0 || error("Reservoir limit must be positive")
    initial_capacity = min(limit, 4096)
    return Reservoir(Matrix{Float32}(undef, INPUT_SIZE, initial_capacity),
                     Vector{Float32}(undef, initial_capacity), 0, 0, limit)
end

function reserve_slot!(reservoir::Reservoir, rng)
    reservoir.seen += 1
    if reservoir.stored < reservoir.limit
        if reservoir.stored == size(reservoir.features, 2)
            new_capacity = min(reservoir.limit, 2 * size(reservoir.features, 2))
            larger_features = Matrix{Float32}(undef, INPUT_SIZE, new_capacity)
            larger_features[:, 1:reservoir.stored] .= @view reservoir.features[:, 1:reservoir.stored]
            resize!(reservoir.labels, new_capacity)
            reservoir.features = larger_features
        end
        reservoir.stored += 1
        return reservoir.stored
    end
    random_index = rand(rng, 1:reservoir.seen)
    return random_index <= reservoir.limit ? random_index : 0
end

function process_daily_files(action, vessels, year_filter)
    files_seen = files_failed = files_missing = 0
    for vessel in vessels, file_path in vessel_files(vessel)
        date = file_date(file_path)
        (date === nothing || !year_filter(year(date))) && continue
        files_seen += 1
        files_seen % 100 == 0 && println("  processed $files_seen files")
        data = try
            read_day(file_path)
        catch error
            files_failed += 1
            @warn "Could not read file; skipping" file_path exception=(error, catch_backtrace())
            continue
        end
        if data === nothing
            files_missing += 1
        else
            action(date, data...)
        end
    end
    println("  files considered=$files_seen, read errors=$files_failed, missing T/flag=$files_missing")
end

function collect_samples(vessels, year_filter; uniform=false, seed=RANDOM_SEED)
    random_generator = MersenneTwister(seed)
    reservoirs = uniform ? (Reservoir(EARLY_STOP_SAMPLE_SIZE),) :
                          (Reservoir(MAX_TRAIN_K), Reservoir(MAX_TRAIN_Z))

    process_daily_files(vessels, year_filter) do date, temperatures, quality_flags
        sample_indices = eligible_indices(temperatures, quality_flags)
        isempty(sample_indices) && return
        day_features = calculate_daily_features(temperatures)
        for sample_index in sample_indices
            label = quality_flags[sample_index] == 'K' ? 1f0 : 0f0
            reservoir = uniform ? reservoirs[1] : reservoirs[label == 1f0 ? 1 : 2]
            slot = reserve_slot!(reservoir, random_generator)
            slot == 0 && continue
            fill_window_features!(@view(reservoir.features[:, slot]), temperatures, day_features, sample_index)
            reservoir.labels[slot] = label
        end
    end

    if uniform
        reservoir = reservoirs[1]
        println("  uniform sample: eligible=$(reservoir.seen), retained=$(reservoir.stored)")
    else
        println("  K eligible/retained = $(reservoirs[1].seen) / $(reservoirs[1].stored)")
        println("  Z eligible/retained = $(reservoirs[2].seen) / $(reservoirs[2].stored)")
    end
    total_samples = sum(reservoir.stored for reservoir in reservoirs)
    features = Matrix{Float32}(undef, INPUT_SIZE, total_samples)
    labels = Matrix{Float32}(undef, 1, total_samples)
    offset = 0
    for reservoir in reservoirs
        destination = (offset + 1):(offset + reservoir.stored)
        features[:, destination] .= @view reservoir.features[:, 1:reservoir.stored]
        labels[1, destination] .= @view reservoir.labels[1:reservoir.stored]
        offset += reservoir.stored
    end
    return features, labels
end

l2_penalty(model) = sum(sum(abs2, parameter) for parameter in Flux.trainables(model))

function mean_focal_loss(model, features, labels)
    sample_count = size(features, 2)
    sample_count == 0 && error("Cannot evaluate loss on an empty dataset")
    total_loss = 0.0
    for batch_start in 1:EVAL_BATCH_SIZE:sample_count
        sample_indices = batch_start:min(batch_start + EVAL_BATCH_SIZE - 1, sample_count)
        loss = focal_loss(model(make_model_input(features, sample_indices)), @view(labels[:, sample_indices]))
        total_loss += Float64(loss) * length(sample_indices)
    end
    return total_loss / sample_count
end

function train_model(model, features, labels, validation_features, validation_labels)
    sample_count = size(features, 2)
    sample_count > 0 || error("No eligible training windows. Check DATA_DIRECTORY, years, T and flags.")
    size(validation_features, 2) > 0 || error("No eligible validation windows for early stopping.")
    0 < sum(labels) < sample_count || error("Training data must contain both K and Z observations.")
    if sum(validation_labels) == 0 || sum(validation_labels) == size(validation_features, 2)
        @warn "Early-stopping sample has only one class; increase EARLY_STOP_SAMPLE_SIZE if possible."
    end
    optimizer = Flux.setup(Adam(LEARNING_RATE), model)
    rng = MersenneTwister(RANDOM_SEED)
    indices = collect(1:sample_count)
    best_loss = Inf
    best_model = deepcopy(model)
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in 1:MAX_EPOCHS
        shuffle!(rng, indices)
        total_loss = 0.0
        for batch_start in 1:BATCH_SIZE:sample_count
            sample_indices = indices[batch_start:min(batch_start + BATCH_SIZE - 1, sample_count)]
            batch_features = make_model_input(features, sample_indices)
            batch_labels = @view labels[:, sample_indices]
            loss, gradients = Flux.withgradient(model) do current_model
                focal_loss(current_model(batch_features), batch_labels) + L2_LAMBDA * l2_penalty(current_model)
            end
            isfinite(loss) || error("Training loss became non-finite in epoch $epoch")
            Flux.update!(optimizer, model, gradients[1])
            total_loss += Float64(loss) * length(sample_indices)
        end
        training_loss = total_loss / sample_count
        validation_loss = mean_focal_loss(model, validation_features, validation_labels)
        isfinite(validation_loss) || error("Validation focal loss became non-finite in epoch $epoch")
        @printf("epoch %3d/%d  train objective %.6f  val focal %.6f", epoch, MAX_EPOCHS, training_loss, validation_loss)
        if validation_loss < best_loss - MIN_VALIDATION_IMPROVEMENT
            best_loss, best_model, best_epoch = validation_loss, deepcopy(model), epoch
            epochs_without_improvement = 0
            println("  * best")
        else
            epochs_without_improvement += 1
            println("  patience $epochs_without_improvement/$EARLY_STOP_PATIENCE")
        end
        if epochs_without_improvement >= EARLY_STOP_PATIENCE
            println("Early stopping; restoring epoch $best_epoch")
            break
        end
    end
    best_epoch > 0 || error("No valid training epoch completed")
    return best_model, best_epoch
end

function predict_scores(model, features)
    sample_count = size(features, 2)
    scores = Vector{Float32}(undef, sample_count)
    for batch_start in 1:EVAL_BATCH_SIZE:sample_count
        sample_indices = batch_start:min(batch_start + EVAL_BATCH_SIZE - 1, sample_count)
        logits = vec(model(make_model_input(features, sample_indices)))
        scores[sample_indices] .= Flux.sigmoid.(logits)
    end
    all(isfinite, scores) || error("Model produced non-finite scores")
    return scores
end

function score_holdout(model, preprocessing, vessels, year_filter)
    scores = Float32[]
    labels = Float32[]
    observed_k = observed_z = 0

    process_daily_files(vessels, year_filter) do date, temperatures, quality_flags
        observed_k += count(==('K'), quality_flags)
        observed_z += count(==('Z'), quality_flags)
        sample_indices = eligible_indices(temperatures, quality_flags)
        isempty(sample_indices) && return
        day_features = calculate_daily_features(temperatures)
        features = Matrix{Float32}(undef, INPUT_SIZE, length(sample_indices))
        for (feature_column, sample_index) in enumerate(sample_indices)
            fill_window_features!(@view(features[:, feature_column]), temperatures, day_features, sample_index)
        end
        scale_features!(features, preprocessing)
        append!(scores, predict_scores(model, features))
        append!(labels, (quality_flags[sample_index] == 'K' ? 1f0 : 0f0 for sample_index in sample_indices))
    end

    eligible_k = Int(sum(labels))
    eligible_z = length(labels) - eligible_k
    println("  eligible observations=$(length(labels)), K=$eligible_k, Z=$eligible_z")
    println("  window coverage: K $eligible_k/$observed_k, Z $eligible_z/$observed_z")
    return (scores=scores, labels=labels, observed_k=observed_k, observed_z=observed_z)
end
ratio(numerator, denominator) = denominator == 0 ? NaN : numerator / denominator

function metrics(scores, labels, threshold)
    true_positives = true_negatives = false_positives = false_negatives = 0
    for sample_index in eachindex(scores, labels)
        actual_k = labels[sample_index] > 0.5f0
        predicted_k = scores[sample_index] >= threshold
        if actual_k && predicted_k
            true_positives += 1
        elseif actual_k
            false_negatives += 1
        elseif predicted_k
            false_positives += 1
        else
            true_negatives += 1
        end
    end
    recall = ratio(true_positives, true_positives + false_negatives)
    precision = ratio(true_positives, true_positives + false_positives)
    specificity = ratio(true_negatives, true_negatives + false_positives)
    return (true_positives=true_positives, false_negatives=false_negatives, false_positives=false_positives, true_negatives=true_negatives,
            recall=recall, precision=precision, false_positive_rate=ratio(false_positives, false_positives + true_negatives),
            specificity=specificity, accuracy=ratio(true_positives + true_negatives, true_positives + true_negatives + false_positives + false_negatives),
            balanced_accuracy=(recall + specificity) / 2,
            f1=ratio(2 * true_positives, 2 * true_positives + false_positives + false_negatives), prevalence=ratio(true_positives + false_negatives, true_positives + false_negatives + false_positives + true_negatives))
end

function threshold_curves(scores, labels; bins=THRESHOLD_BINS)
    positive_bins = zeros(Int, bins + 1)
    negative_bins = zeros(Int, bins + 1)
    for threshold_index in eachindex(scores, labels)
        bin_index = clamp(floor(Int, Float64(scores[threshold_index]) * bins), 0, bins) + 1
        if labels[threshold_index] > 0.5f0
            positive_bins[bin_index] += 1
        else
            negative_bins[bin_index] += 1
        end
    end
    positive_count, negative_count = sum(positive_bins), sum(negative_bins)
    thresholds = Float64[nextfloat(1.0)]
    recall = Float64[positive_count > 0 ? 0.0 : NaN]
    false_positive_rate = Float64[negative_count > 0 ? 0.0 : NaN]
    precision = Float64[1.0]
    f1 = Float64[positive_count > 0 ? 0.0 : NaN]
    true_positives = false_positives = 0
    for bin_index in bins:-1:0
        true_positives += positive_bins[bin_index + 1]
        false_positives += negative_bins[bin_index + 1]
        push!(thresholds, bin_index / bins)
        push!(recall, ratio(true_positives, positive_count))
        push!(false_positive_rate, ratio(false_positives, negative_count))
        push!(precision, true_positives + false_positives == 0 ? 1.0 : true_positives / (true_positives + false_positives))
        push!(f1, ratio(2 * true_positives, positive_count + true_positives + false_positives))
    end
    roc_auc = positive_count > 0 && negative_count > 0 ?
        sum((false_positive_rate[threshold_index] - false_positive_rate[threshold_index-1]) * (recall[threshold_index] + recall[threshold_index-1]) / 2 for threshold_index in 2:length(false_positive_rate)) : NaN
    average_precision = positive_count > 0 ?
        sum((recall[threshold_index] - recall[threshold_index-1]) * precision[threshold_index] for threshold_index in 2:length(recall)) : NaN
    return (thresholds=thresholds, recall=recall, false_positive_rate=false_positive_rate, precision=precision,
            f1=f1, roc_auc=roc_auc, average_precision=average_precision,
            positive_count=positive_count, negative_count=negative_count)
end

function choose_threshold(curves)
    curves.positive_count > 0 || error("No K observations in full validation; cannot choose threshold.")
    curves.negative_count > 0 || error("No Z observations in full validation; cannot choose threshold.")
    if THRESHOLD_MODE == :f1
        threshold_index = argmax(curves.f1)
    elseif THRESHOLD_MODE == :max_recall_at_fpr
        0 <= MAX_VALIDATION_FPR <= 1 || error("MAX_VALIDATION_FPR must be between 0 and 1")
        candidates = findall(rate -> rate <= MAX_VALIDATION_FPR, curves.false_positive_rate)

        threshold_index = candidates[argmax(curves.recall[candidates])]
    else
        error("Unknown THRESHOLD_MODE; use :f1 or :max_recall_at_fpr")
    end
    return curves.thresholds[threshold_index]
end

function print_metrics(title, performance, threshold)
    println("\n$title")
    @printf("  threshold            %.4f\n", threshold)
    @printf("  K recall             %8.2f %%\n", 100 * performance.recall)
    @printf("  K precision          %8.2f %%\n", 100 * performance.precision)
    @printf("  Z false-positive rate %7.2f %%\n", 100 * performance.false_positive_rate)
    @printf("  specificity          %8.2f %%\n", 100 * performance.specificity)
    @printf("  accuracy             %8.2f %%\n", 100 * performance.accuracy)
    @printf("  balanced accuracy    %8.2f %%\n", 100 * performance.balanced_accuracy)
    @printf("  F1                   %8.2f %%\n", 100 * performance.f1)
    @printf("  eligible K prevalence %7.2f %%\n", 100 * performance.prevalence)
    println("  TP/FN = $(performance.true_positives)/$(performance.false_negatives); FP/TN = $(performance.false_positives)/$(performance.true_negatives)")
end

function append_metrics_csv(split, performance, threshold, curves, data)
    file_path = joinpath(OUTPUT_DIRECTORY, "metrics.csv")
    open(file_path, "a") do output_file
        @printf(output_file, "%s,%.8f,%d,%d,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%d,%d,%.8f,%d,%d,%.8f\n",
            split, threshold, performance.true_positives, performance.false_negatives, performance.false_positives, performance.true_negatives, performance.recall, performance.precision,
            performance.false_positive_rate, performance.specificity, performance.accuracy, performance.balanced_accuracy, performance.f1,
            performance.prevalence, curves.roc_auc, curves.average_precision,
            performance.true_positives + performance.false_negatives, data.observed_k, ratio(performance.true_positives + performance.false_negatives, data.observed_k),
            performance.false_positives + performance.true_negatives, data.observed_z, ratio(performance.false_positives + performance.true_negatives, data.observed_z))
    end
end

function report_split(data, name, title, threshold)
    if isempty(data.scores)
        println("No eligible observations for $title")
        return
    end
    performance = metrics(data.scores, data.labels, threshold)
    curves = threshold_curves(data.scores, data.labels)
    print_metrics(title, performance, threshold)
    @printf("  ROC AUC ≈ %.4f; average precision ≈ %.4f (0.001 score bins)\n", curves.roc_auc, curves.average_precision)
    append_metrics_csv(name, performance, threshold, curves, data)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
