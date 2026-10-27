using Downloads
using Base.Threads

const HOST = "https://tds.coaps.fsu.edu"

const VESSELS = Dict(
    "WTEO" => 2007:2026,
    "WTER" => 2007:2026,
    "WTEE" => 2009:2026,
    "WTEA" => 2012:2026,
    "VNAA" => 2008:2020,
)

function getpage(url::String)
    tmp = tempname()

    try
        Downloads.download(url, tmp)
        return read(tmp, String)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
end

function filenames_for_year(callsign::String, year::Int)
    catalog_url =
        "$HOST/thredds/catalog/samos/data/research/$callsign/$year/catalog.html"

    println("Reading catalog: $catalog_url")

    html = try
        getpage(catalog_url)
    catch err
        @warn "Could not read catalog" callsign year exception=err
        return String[]
    end

    pattern = Regex(callsign * raw"_\d{8}v\d+\.nc")

    filenames = unique(m.match for m in eachmatch(pattern, html))
    sort!(filenames)

    return filenames
end

function download_file(
    callsign::AbstractString,
    year::Int,
    filename::AbstractString,
    output_dir::AbstractString
)

    url =
        "$HOST/thredds/fileServer/samos/data/research/" *
        "$callsign/$year/$filename"

    destination = joinpath(output_dir, filename)

    if isfile(destination)
        return :skipped
    end

    try
        Downloads.download(url, destination)

        println(
            "[thread $(threadid())] downloaded $filename"
        )

        return :downloaded

    catch err
        @warn "Download failed" url exception=err

        isfile(destination) && rm(destination; force=true)

        return :failed
    end
end

function download_vessel(callsign::String, years)

    output_dir = "$(callsign)-research-q"
    mkpath(output_dir)

    println()
    println("="^70)
    println("VESSEL: $callsign")
    println("OUTPUT: $output_dir")
    println("THREADS: $(nthreads())")
    println("="^70)

    total_downloaded = Atomic{Int}(0)
    total_skipped    = Atomic{Int}(0)
    total_failed     = Atomic{Int}(0)

    for year in years

        filenames = filenames_for_year(callsign, year)

        println(
            "[$callsign / $year] found $(length(filenames)) files"
        )

        Threads.@threads for i in eachindex(filenames)

            filename = filenames[i]

            result = download_file(
                callsign,
                year,
                filename,
                output_dir
            )

            if result === :downloaded
                atomic_add!(total_downloaded, 1)

            elseif result === :skipped
                atomic_add!(total_skipped, 1)

            elseif result === :failed
                atomic_add!(total_failed, 1)
            end
        end
    end

    println()
    println("$callsign finished")
    println("  downloaded  : $(total_downloaded[])")
    println("  already had : $(total_skipped[])")
    println("  failed      : $(total_failed[])")
end

function main()

    println("SAMOS research-quality downloader")
    println("Julia threads available: $(nthreads())")

    for (callsign, years) in VESSELS
        download_vessel(callsign, years)
    end

    println()
    println("All vessels finished.")
end

main()