using Statistics

function preprocess_sim(dat; bg = 100f0, num_directions=nothing)
    dat = dat.-bg # copies
    dat .*= mean(dat) ./ mean(dat, dims=(1:ndims(dat)-1))
    if !isnothing(num_directions)
        all_mean =  mean(dat, dims=ndims(dat))
        num_phases = size(dat, ndims(dat)) ÷ num_directions
        for d in 1:num_directions
            sub_data = slice(dat, ndims(dat), (d-1)*num_phases+1:d*num_phases)
            sub_mean = mean(sub_data, dims=ndims(sub_data))
            sub_data .*= all_mean .* sub_mean ./ (abs2.(sub_mean) .+ 1e-6)
        end
    end
    return dat
end
