

module TripletModule
#本模块用于度量学习中三元组的相关操作
using LinearAlgebra, Distributed
using NearestNeighbors

export Triplet, build_triplets

max_triplet_number = 5000
noise_num = 5

mutable struct Triplet
    xi::Array{Float32,1}
    xj::Array{Float32,1}
    xk::Array{Float32,1}
    ij_dis::Float32
    jk_dis::Float32
    weight::Float32
end

# 根据数据的特征和标签构建三元组
function build_triplets(features::Array{Float32,2}, labels)
    triplets::Array{Triplet,1} = []
    trs = Dict()  # 字典，存放了标签及其对应的数据子集
    for la in unique(labels)
        inds = labels .== la
        trs[la] = features[inds, :]
    end
    for (la, xs_buck) in trs
        # 首先求解同标签最近数据点
        kdt_i = KDTree(xs_buck')
        # 如果某类别的样本数小于k，则跳过
        ss = size(xs_buck)
        if ss[1] < 2
            continue
        end
        indices1, dists1 = knn(kdt_i, xs_buck', 2, true)
        xts = nothing
        # 搜索异类最近的数据点
        # 将所有的异类数据拼接
        for (lp, xt_buck) in trs
            # 说明是同类，跳过
            if lp == la
                continue
            elseif xts == nothing
                xts = xt_buck
            else
                xts = cat(xts, xt_buck, dims = 1)
            end
        end
        kdt_t = KDTree(xts')
        indices2, dists2 = knn(kdt_t, xs_buck', 1, true)
        ss = size(xs_buck)
        for i = 1:ss[1]
            try
                xj = xs_buck[i, :]
                xi = xs_buck[indices1[i][2], :]
                ij_dis = dists1[i][2]
                xt = xts[indices2[i][1],:]
                jt_dis = dists2[i][1]
                triplet = Triplet(xi, xj, xt, ij_dis, jt_dis,1.0)
                triplet.weight = compute_triplet_weight(triplet)
                push!(triplets, triplet)
            catch e
                # println(e)
            end
        end
    end
    if length(triplets) > noise_num
        sort!(triplets,by= t -> (t.ij_dis-t.jk_dis), rev = true)
        triplets = triplets[noise_num+1:end]
        # println([t.weight for t in triplets])
    end
    if length(triplets) > max_triplet_number
        sort!(triplets,by= t -> t.weight, rev = true)
        triplets = triplets[1:max_triplet_number]
        # println([t.weight for t in triplets])
    end
    return triplets
end

function compute_triplet_weight(trip::Triplet)
    rou = 1.0/4.5
    gap = abs(trip.ij_dis-trip.jk_dis)
    w = exp(-gap/rou)
    return w
end



end
