const MODEL_TITLE = "Multi-Scale Unsupervised CNN Denoising Autoencoder (Daily 4-Hour Block Mode)"
const OUTPUT_NAME = "met_autoencoder_rolling_results"

using NCDatasets
using Flux
using Statistics
using Random
using Dates
using Printf
using Serialization
using Plots

# ==============================================================================
# Configuration & Constants
# ==============================================================================

const DATA_DIRECTORY = get(ENV, "SAMOS_ROOT", ".")
const VESSELS = ["WTEA", "WTEE", "WTER"]
const TRAIN_END_YEAR = 2022
const VALIDATION_YEAR = 2023
const TEST_YEAR = 2024

# Time Series Window Configuration (1-minute sampling rate)
const WINDOW_SIZE = 240       # 4 Hours (240 minutes) per block
const BLOCKS_PER_FILE = 6     # 6 blocks * 240 mins = 1440 mins (1 Day)
const NUM_VARIABLES = 2       # Temperature (T) and Relative Humidity (RH)
const LATENT_DIM = 32         # Latent space dimension

# Hyperparameters & Optimization
const LEARNING_RATE = 1f-3
const L2_LAMBDA = 1f-5
const NOISE_FACTOR = 0.1f0  
const BATCH_SIZE = 64       
const MAX_EPOCHS = 10
const EARLY_STOP_PATIENCE = 3
const ROLLING_WINDOW_DAYS = 7
const Z_THRESHOLD = 3.5f0   

const OUTPUT_DIRECTORY = joinpath(DATA_DIRECTORY, OUTPUT_NAME)

# ==============================================================================
# Utility & Data Quality Functions
# ==============================================================================

function flag_character(value)::Char
    ismissing(value) && return '?'
    value isa Char && return value
    value isa Integer && return Char(value)
    value isa AbstractString && return isempty(value) ? '?' : first(value)
    return first(string(value))
end

function get_quality_flags(dataset)
    flag_variable = dataset["flag"]
    flag_data = Array(flag_variable)
    dim_names = String.(NCDatasets.dimnames(flag_variable))
    time_dim = findfirst(==("time"), dim_names)
    flag_idx = haskey(dataset["T"].attrib, "qcindex") ? Int(dataset["T"].attrib["qcindex"]) : 1
    
    quality_flags = if time_dim == 1
        flag_data[:, flag_idx]
    elseif time_dim == 2
        flag_data[flag_idx, :]
    else
        flag_data[:, 1]
    end
    return flag_character.(vec(quality_flags))
end

# ==============================================================================
# Neural Network Architecture
# ==============================================================================

struct MultiScaleEncoder{S<:Chain, M<:Chain, L<:Chain, D<:Dense}
    conv_short::S
    conv_med::M
    conv_long::L
    dense_latent::D
end

Flux.@layer MultiScaleEncoder

function MultiScaleEncoder(in_channels::Int, window_size::Int, latent_dim::Int)
    conv_short = Chain(Conv((5,), in_channels => 16, relu; pad=2), MaxPool((4,)))
    conv_med   = Chain(Conv((15,), in_channels => 16, relu; pad=7), MaxPool((4,)))
    conv_long  = Chain(Conv((31,), in_channels => 16, relu; pad=15), MaxPool((4,)))
    
    flattened_size = 16 * (window_size ÷ 4) * 3 
    dense_latent = Dense(flattened_size => latent_dim)
    return MultiScaleEncoder(conv_short, conv_med, conv_long, dense_latent)
end

function (m::MultiScaleEncoder)(x)
    f_short = m.conv_short(x)
    f_med   = m.conv_med(x)
    f_long  = m.conv_long(x)
    combined = cat(f_short, f_med, f_long, dims=2)
    flattened = Flux.flatten(combined)
    return m.dense_latent(flattened)
end

function make_decoder(latent_dim::Int, window_size::Int, out_channels::Int)
    return Chain(
        Dense(latent_dim => 32 * (window_size ÷ 4), relu),
        x -> reshape(x, window_size ÷ 4, 32, :),
        Upsample(4, :nearest), 
        Conv((15,), 32 => 16, relu; pad=7),
        Conv((5,), 16 => out_channels; pad=2) 
    )
end

function make_autoencoder()
    encoder = MultiScaleEncoder(NUM_VARIABLES, WINDOW_SIZE, LATENT_DIM)
    decoder = make_decoder(LATENT_DIM, WINDOW_SIZE, NUM_VARIABLES)
    return Chain(encoder, decoder)
end

function denoising_mse_loss(model, clean_inputs)
    noise = randn(Float32, size(clean_inputs)) .* NOISE_FACTOR
    noisy_inputs = clean_inputs .+ noise
    reconstructions = model(noisy_inputs)
    return mean((reconstructions .- clean_inputs) .^ 2)
end

# ==============================================================================
# Streaming Anomaly Tracker
# ==============================================================================

mutable struct VesselAnomalyTracker
    vessel_name::String
    window_size_steps::Int
    error_history::Vector{Float32}
    z_threshold::Float32
end

function VesselAnomalyTracker(vessel_name::String, window_days::Int, sampling_rate_mins::Int=1; z_threshold::Float32=Z_THRESHOLD)
    steps = (window_days * 24 * 60) ÷ (WINDOW_SIZE * sampling_rate_mins)
    return VesselAnomalyTracker(vessel_name, steps, Float32[], z_threshold)
end

function update_and_score!(tracker::VesselAnomalyTracker, raw_mse::Float32)
    push!(tracker.error_history, raw_mse)
    if length(tracker.error_history) > tracker.window_size_steps
        popfirst!(tracker.error_history)
    end
    if length(tracker.error_history) < max(2, tracker.window_size_steps ÷ 10)
        return 0.0f0, false
    end
    local_mean = mean(tracker.error_history)
    local_std  = max(std(tracker.error_history; corrected=false), 1f-5)
    rolling_z = (raw_mse - local_mean) / local_std
    return rolling_z, (rolling_z >= tracker.z_threshold)
end

# ==============================================================================
# Data Parsing & Preprocessing
# ==============================================================================

function read_day_multichannel(file_path::AbstractString)
    return NCDataset(file_path, "r") do dataset
        if !haskey(dataset, "T") || !haskey(dataset, "RH") || !haskey(dataset, "flag")
            return nothing
        end
        raw_T = vec(Array(dataset["T"]))
        raw_RH = vec(Array(dataset["RH"]))
        quality_flags = get_quality_flags(dataset)
        
        sample_count = min(length(raw_T), length(raw_RH), length(quality_flags))
        sensor_matrix = fill(NaN32, sample_count, NUM_VARIABLES)
        clean_flags = Vector{Char}(undef, sample_count)
        
        for i in 1:sample_count
            val_t = Float32(raw_T[i])
            val_rh = Float32(raw_RH[i])
            if isfinite(val_t) && -60 < val_t < 60 && isfinite(val_rh) && 0 <= val_rh <= 100
                sensor_matrix[i, 1] = val_t
                sensor_matrix[i, 2] = val_rh
            end
            clean_flags[i] = quality_flags[i]
        end
        return sensor_matrix, clean_flags
    end
end

function vessel_files(vessel::AbstractString)
    vessel_directory = joinpath(DATA_DIRECTORY, "$(vessel)-research-q")
    if !isdir(vessel_directory)
        @warn "Missing vessel directory" vessel vessel_directory
        return String[]
    end
    files = readdir(vessel_directory; join=true)
    return sort(filter(f -> endswith(lowercase(f), ".nc"), files))
end

function file_date(file_path::AbstractString)
    date_match = match(r"_(\d{4})(\d{2})(\d{2})v", basename(file_path))
    date_match === nothing && return nothing
    return Date(
        parse(Int, date_match.captures[1]),
        parse(Int, date_match.captures[2]),
        parse(Int, date_match.captures[3])
    )
end

function extract_windows(vessels, year_filter)
    all_windows = Vector{Array{Float32, 2}}()
    all_labels = Vector{Float32}() 
    
    for vessel in vessels, file_path in vessel_files(vessel)
        date = file_date(file_path)
        (date === nothing || !year_filter(year(date))) && continue
        
        data = try 
            read_day_multichannel(file_path) 
        catch
            continue
        end
        data === nothing && continue
        sensor_matrix, flags = data
        
        total_file_rows = size(sensor_matrix, 1)
        
        for block_idx in 1:BLOCKS_PER_FILE
            start_idx = (block_idx - 1) * WINDOW_SIZE + 1
            end_idx   = block_idx * WINDOW_SIZE
            
            block_data = zeros(Float32, WINDOW_SIZE, NUM_VARIABLES)
            block_flag = 0f0
            
            avail_start = start_idx
            avail_end = min(end_idx, total_file_rows)
            
            if avail_start <= total_file_rows
                slice_len = avail_end - avail_start + 1
                raw_slice = sensor_matrix[avail_start:avail_end, :]
                
                raw_slice[isnan.(raw_slice)] .= 0f0
                
                s_mean = mean(raw_slice, dims=1)
                s_std  = max.(std(raw_slice, dims=1; corrected=false), 1f-3)
                normalized_slice = (raw_slice .- s_mean) ./ s_std
                
                block_data[1:slice_len, :] .= normalized_slice
                block_flag = flags[avail_end] == 'K' ? 1f0 : 0f0
            end
            
            push!(all_windows, block_data)
            push!(all_labels, block_flag)
        end
    end
    
    n_windows = length(all_windows)
    if n_windows == 0
        return Array{Float32, 3}(undef, WINDOW_SIZE, NUM_VARIABLES, 0), Float32[]
    end
    
    # Efficient 3D tensor pre-allocation (avoids heavy cat(...) splatting allocations)
    packed_windows = Array{Float32, 3}(undef, WINDOW_SIZE, NUM_VARIABLES, n_windows)
    for i in 1:n_windows
        packed_windows[:, :, i] = all_windows[i]
    end
    
    return packed_windows, all_labels
end

# ==============================================================================
# Model Training Engine
# ==============================================================================

function train_pipeline(model, train_x, val_x)
    sample_count = size(train_x, 3)
    optimizer = Flux.setup(Adam(LEARNING_RATE), model)
    indices = collect(1:sample_count)
    best_loss = Inf
    best_model = deepcopy(model)
    epochs_without_improvement = 0
    
    train_curve = Float64[]
    val_curve = Float64[]
    
    l2_penalty(m) = sum(sum(abs2, p) for p in Flux.params(m))

    for epoch in 1:MAX_EPOCHS
        shuffle!(indices)
        total_loss = 0.0
        
        for batch_start in 1:BATCH_SIZE:sample_count
            batch_end = min(batch_start + BATCH_SIZE - 1, sample_count)
            batch_indices = indices[batch_start:batch_end]
            batch_inputs = train_x[:, :, batch_indices]
            
            loss, gradients = Flux.withgradient(model) do m
                denoising_mse_loss(m, batch_inputs) + L2_LAMBDA * l2_penalty(m)
            end
            Flux.update!(optimizer, model, gradients[1])
            total_loss += Float64(loss) * length(batch_indices)
        end
        
        train_loss = total_loss / sample_count
        val_loss = Float64(mean(denoising_mse_loss(model, val_x)))
        
        push!(train_curve, train_loss)
        push!(val_curve, val_loss)
        
        @printf("Epoch %3d/%d | Train MSE: %.6f | Val MSE: %.6f\n", epoch, MAX_EPOCHS, train_loss, val_loss)
        
        if val_loss < best_loss
            best_loss = val_loss
            best_model = deepcopy(model)
            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
        end
        
        if epochs_without_improvement >= EARLY_STOP_PATIENCE
            println("Early stopping triggered at epoch $epoch.")
            break
        end
    end
    
    return best_model, train_curve, val_curve
end

# ==============================================================================
# Metrics & Evaluation
# ==============================================================================

function evaluate_metrics(predictions, ground_truths)
    tp = fn = fp = tn = 0
    for (p, g) in zip(predictions, ground_truths)
        is_true = g > 0.5f0
        if is_true && p
            tp += 1
        elseif is_true && !p
            fn += 1
        elseif !is_true && p
            fp += 1
        else
            tn += 1
        end
    end
    
    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return (tp=tp, fn=fn, fp=fp, tn=tn, recall=recall, precision=precision, accuracy=accuracy, f1=f1)
end

# ==============================================================================
# Main Program Flow
# ==============================================================================

function main()
    mkpath(OUTPUT_DIRECTORY)
    
    println("\nExtracting Training Windows (thru $TRAIN_END_YEAR)...")
    train_x, _ = extract_windows(VESSELS, y -> y <= TRAIN_END_YEAR)
    
    println("Extracting Validation Windows ($VALIDATION_YEAR)...")
    val_x, _ = extract_windows(VESSELS, y -> y == VALIDATION_YEAR)
    
    if size(train_x, 3) == 0 || size(val_x, 3) == 0
        error("Extraction layer returned 0 blocks. Ensure your data path is populated correctly.")
    end
    
    println("\nInitializing Autoencoder Module...")
    model = make_autoencoder()
    trained_model, train_loss, val_loss = train_pipeline(model, train_x, val_x)
    
    serialize(joinpath(OUTPUT_DIRECTORY, "trained_autoencoder.jls"), trained_model)
    println("\n[SAVED] Serialized model exported to output path.")
    
    p_curves = plot(
        train_loss, label="Train Loss", title="Autoencoder Learning",
        xlabel="Epochs", ylabel="MSE Loss", linewidth=2
    )
    plot!(p_curves, val_loss, label="Validation Loss", linewidth=2)
    savefig(p_curves, joinpath(OUTPUT_DIRECTORY, "learning_curves.png"))
    
    for vessel in VESSELS
        println("\nStreaming Inference over $vessel for Test Horizon ($TEST_YEAR)...")
        test_x, test_labels = extract_windows([vessel], y -> y == TEST_YEAR)
        size(test_x, 3) == 0 && continue
        
        tracker = VesselAnomalyTracker(vessel, ROLLING_WINDOW_DAYS)
        reconstructions = trained_model(test_x)
        num_samples = size(test_x, 3)
        raw_mses = Vector{Float32}(undef, num_samples)
        pred_flags = Vector{Bool}(undef, num_samples)
        
        for t in 1:num_samples
            raw_mses[t] = mean((reconstructions[:, :, t] .- test_x[:, :, t]) .^ 2)
            _, pred_flags[t] = update_and_score!(tracker, raw_mses[t])
        end
        
        stats = evaluate_metrics(pred_flags, test_labels)
        @printf("\n[%s Test Set Metrics Alignment]\n", vessel)
        @printf("  Accuracy  : %.2f%%\n", stats.accuracy * 100)
        @printf("  Precision : %.2f%%\n", stats.precision * 100)
        @printf("  Recall    : %.2f%%\n", stats.recall * 100)
        @printf("  F1-Score  : %.2f\n", stats.f1)
        
        worst_idx = argmax(raw_mses)
        best_idx  = argmin(raw_mses)
        
        p_worst = plot(
            test_x[:, 1, worst_idx], label="True T",
            title="Worst 4-Hour Block Profile (Anomaly)",
            xlabel="Minute Step", ylabel="Normalized Scale"
        )
        plot!(p_worst, test_x[:, 2, worst_idx], label="True RH")
        plot!(p_worst, reconstructions[:, 1, worst_idx], label="Recon T", linestyle=:dash)
        plot!(p_worst, reconstructions[:, 2, worst_idx], label="Recon RH", linestyle=:dash)
        savefig(p_worst, joinpath(OUTPUT_DIRECTORY, "$(vessel)_worst_reconstruction.png"))
        
        p_best = plot(
            test_x[:, 1, best_idx], label="True T",
            title="Best 4-Hour Block Profile (Normal)",
            xlabel="Minute Step", ylabel="Normalized Scale"
        )
        plot!(p_best, test_x[:, 2, best_idx], label="True RH")
        plot!(p_best, reconstructions[:, 1, best_idx], label="Recon T", linestyle=:dash)
        plot!(p_best, reconstructions[:, 2, best_idx], label="Recon RH", linestyle=:dash)
        savefig(p_best, joinpath(OUTPUT_DIRECTORY, "$(vessel)_best_reconstruction.png"))
    end
    
    println("\nPipeline finished successfully. Outputs saved to: $OUTPUT_DIRECTORY")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end