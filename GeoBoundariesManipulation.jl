module GeoBoundariesManipulation

    export inanypolygon, load_boundaries, save_to_txt, overall_plot

    using PolygonOps
    using DelimitedFiles
    using DataFrames
    using Plots
    using StatsBase
    using LinearAlgebra
    loop_ended = false
    file_list = readdir(joinpath(pwd(),"boundaries"))

    function boundaryOutput(filelist,nx,ny)

        function optimas(filelist) #Find max and mins to use in VectorOfPolygons/Squares

            polyg_data1 = readdlm(string(pwd(),"\\boundaries\\",filelist[1]))
            max1 = maximum(polyg_data1[1:end,1])
            min1 = minimum(polyg_data1[1:end,1])
            max2 = maximum(polyg_data1[1:end,2])
            min2 = minimum(polyg_data1[1:end,2])
        
            for file in filelist[2:end]
                polyg_data = readdlm(string(pwd(),"\\boundaries\\",file))
                p1_max = maximum(polyg_data[1:end,1])
                p1_min = minimum(polyg_data[1:end,1])
                p2_max = maximum(polyg_data[1:end,2])
                p2_min = minimum(polyg_data[1:end,2])
        
                if p1_max > max1
                    max1 = p1_max
                end
        
                if p1_min < min1
                    min1 = p1_min
                end
        
                if p2_max > max2
                    max2 = p2_max
                end
        
                if p2_min < min2
                    min2 = p2_min
                end
            end
            return max1, min1, max2, min2
        end

        function VectorOfAllPolygons(filelist) 
            max1 = optimas(filelist)[1]
            min1 = optimas(filelist)[2]
            max2 = optimas(filelist)[3]
            min2 = optimas(filelist)[4]
        
            lmbda = (max1-min1)/(max2-min2)
            eps = 0.0001
            offset = 0.5 - lmbda/2
        
            allpolygons = Vector{Tuple{Float64,Float64}}[]
        
            for file in filelist
                polyg_data = readdlm(string(pwd(),"\\boundaries\\",file))
        
                polygon = Tuple{Float64,Float64}[]
                for i in 1:length(polyg_data[1:end,1])
                    if polyg_data[i,1] == min1 
                        push!(polygon,(lmbda*eps+offset, (polyg_data[i,2]-min2)/(max2-min2)))
                    elseif polyg_data[i,1] == max1
                        push!(polygon,(lmbda*(1-eps)+offset, (polyg_data[i,2]-min2)/(max2-min2)))
                    elseif polyg_data[i,2] == min2
                        push!(polygon,(lmbda*((polyg_data[i,1]-min1)/(max1-min1))+offset, eps))
                    elseif polyg_data[i,2] == max2
                        push!(polygon,(lmbda*((polyg_data[i,1]-min1)/(max1-min1))+offset, 1-eps))
                    else
                        push!(polygon,(lmbda*((polyg_data[i,1]-min1)/(max1-min1))+offset, (polyg_data[i,2]-min2)/(max2-min2)))
                    end
                end
        
                push!(allpolygons,polygon)
        
            end
            return allpolygons
        end
        
        function VectorOfAllSquares(pol,nx,ny)
        
            function pol_points_to_squares_vector()
        
                all_square_boundaries = Vector{Tuple{Float64,Float64}}[]
        
                for k in 1:length(pol)
                    square_boundary = Tuple{Float64,Float64}[]
                    #Normalised square points of each point
                    for i in 1:length(pol[k])
                        x_max = round(pol[k][i][1], digits=1, base=nx, RoundUp)
                        x_min = round(pol[k][i][1], digits=1, base=nx, RoundDown)
                        y_max = round(pol[k][i][2], digits=1, base=ny, RoundUp)
                        y_min = round(pol[k][i][2], digits=1, base=ny, RoundDown)
                        
                        push!(square_boundary,(x_min,y_max),(x_min,y_min),(x_max,y_min),(x_max,y_max))
                    end
                    #union(square_boundary)
                    push!(all_square_boundaries,square_boundary)
                end
        
                return all_square_boundaries
            end
        
            function delete_in_polygon(vector_of_points,polygon)
                for i in length(vector_of_points):-1:1
                    if abs(inpolygon(vector_of_points[i],polygon)) == 1
                        deleteat!(vector_of_points,i)
                    end
                end
                return vector_of_points
            end
            
            #---Function Output
            VecOfSquares = pol_points_to_squares_vector()
            
            for i in 1:length(VecOfSquares)
                eliminate_duplicates(VecOfSquares[i],1)
                delete_in_polygon(VecOfSquares[i],pol[i])
            end    
        
            return VecOfSquares
        end

        # MAIN
        polygons = VectorOfAllPolygons(filelist)
        squares = VectorOfAllSquares(polygons,nx,ny)
        boundaries = auto_boundaries(squares,polygons)

        println("Do you want to check out the boundary plots? [Y/N]: ")
        see_plots = readline()
        if see_plots == "Y"
            for i in 1:length(polygons)
                display(check_plot(boundaries[i], polygons[i],nx,ny))
            end
        elseif see_plots =="N"
            println("Proceeding to finalising boundaries.")
        else
            println("Wrong input.")
        end

        println("Would you like to proceed to manual selection? [Y/N]")
        manual_auto_decision = readline()
        if manual_auto_decision == "Y"
            squares = VectorOfAllSquares(polygons,nx,ny)
            boundaries = manual_boundaries(squares,polygons)
            final_boundaries = auto_boundaries(boundaries,polygons)
            save_to_txt(final_boundaries,"finalboundaries")
            return final_boundaries
        elseif manual_auto_decision == "N"
            save_to_txt(boundaries,"finalboundaries")
            return boundaries
        else
            println("Wrong input. You will have to run again the function.")
        end 
        

    end

    function optimas(filelist) #Find max and mins to use in VectorOfPolygons/Squares

        polyg_data1 = readdlm(joinpath(pwd(),"boundaries",filelist[1]))
        max1 = maximum(polyg_data1[1:end,1])
        min1 = minimum(polyg_data1[1:end,1])
        max2 = maximum(polyg_data1[1:end,2])
        min2 = minimum(polyg_data1[1:end,2])

        for file in filelist[2:end]
            polyg_data = readdlm(joinpath(pwd(),"boundaries",file))
            p1_max = maximum(polyg_data[1:end,1])
            p1_min = minimum(polyg_data[1:end,1])
            p2_max = maximum(polyg_data[1:end,2])
            p2_min = minimum(polyg_data[1:end,2])

            if p1_max > max1
                max1 = p1_max
            end

            if p1_min < min1
                min1 = p1_min
            end

            if p2_max > max2
                max2 = p2_max
            end

            if p2_min < min2
                min2 = p2_min
            end
        end
        return max1, min1, max2, min2
    end

    function eliminate_duplicates(arr,k)
        i = 1;  
        freq = countmap(arr)
        while i <= length(arr)
            if freq[arr[i]] > k
                arr = deleteat!(arr,i)
                i-=1;
            end
            i+=1;
            freq = countmap(arr)
        end
        return arr     
    end

    function sort_points(sq)
        
        sq = eliminate_duplicates(sq,1) #eliminate_duplicates
        sq_new = [sq[1]]
        deleteat!(sq,1)
        
        while isempty(sq) == false
            d = []    
            for i in 1:length(sq)
                push!(d, sqrt((sq[i][1]-sq_new[length(sq_new)][1])^2 + (sq[i][2]-sq_new[length(sq_new)][2])^2))
            end
            min_distance_index = findmin(d)[2]
            push!(sq_new,sq[min_distance_index])
            deleteat!(sq,min_distance_index)
        end

        return sq_new
    end

    function auto_boundaries(sq,pol)
        global file_list
        boundaries = Vector{Tuple{Float64,Float64}}[]

        function sorting(sq,polygon)

            function intersection(p1,p2,b1,b2)
                b_x = (b1[1]-p1[1])
                b_y = (b1[2]-p1[2])
                A_11 = (p2[1]-p1[1])
                A_12 = (p2[2]-p1[2])
                A_21 = (b1[1]-b2[1])
                A_22 = (b1[2]-b2[2])
            
                b = [b_x;b_y]
                A = [A_11 A_21;A_12 A_22]
            
                x = [10;10]
                try
                    x = lu(A)\b
                catch 
                    x = [10;10]
                end
            
                
            
                if (0<=x[1]) && (x[1]<=1) && (0<=x[2]) && (x[2]<=1) 
                    return true
                else
                    return false
                end
            
            end

            function sort_points_inter(sq,polygon)
            
                sq = eliminate_duplicates(sq,1)
                sq_new = [sq[1]]
                deleteat!(sq,1)
                
                while isempty(sq) == false
            
                    d = []  
                    for i in 1:length(sq)
                        push!(d, sqrt((sq[i][1]-sq_new[length(sq_new)][1])^2 + (sq[i][2]-sq_new[length(sq_new)][2])^2))
                    end
            
                    min_distance_index = findmin(d)[2]
                    point_a_bound = sq[min_distance_index]
                    point_b_bound = last(sq_new)
                    
                    intersection_check = false
                    for i in 1:(length(polygon)-1) # Find a way to pass island vector (vec)
                        point_a_pol = polygon[i]
                        point_b_pol = polygon[i+1]
                        intersection_check = intersection(point_a_bound,point_b_bound,point_a_pol,point_b_pol)
                        if intersection_check == true
                            deleteat!(sq,min_distance_index)
                            break
                        end
                    end
            
                    if intersection_check == false
                        push!(sq_new,sq[min_distance_index])
                        deleteat!(sq,min_distance_index)
                    end
                end
                
                return sq_new
            end

            function remove_consecutive_collinears(sq)

                function collinear(a,b,c)
                    det = a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2])
                    solution_threshold = 10^(-4)
                
                    if det < solution_threshold && (a[1]==b[1]==c[1] || a[2]==b[2]==c[2])
                        return true
                    else
                        return false
                    end
                end

                collinears = []
                for i in 1:(length(sq)-2)
                    if collinear(sq[i],sq[i+1],sq[i+2])
                        push!(collinears,i+1)
                        j = 1
                        if i+2+j < length(sq)               
                            while collinear(sq[i],sq[i+1+j],sq[i+2+j])
                                push!(collinears,i+1+j)
                                j += 1
                                if i+2+j > length(sq)
                                    break
                                end
                            end
                        end
                        #delete action
                    end

                end

                eliminate_duplicates(collinears,1)
                collinears = sort(collinears, rev=true)
                for i in 1:(length(collinears))
                    deleteat!(sq,collinears[i])
                end

                return sq
            end

            function add_last_point(sq_new) 
                global loop_ended
                if sq_new[1] != last(sq_new)    
                    #global loop_ended       
                    while loop_ended == false
                        #global loop_ended
                        loop_ended = true
                        for i in 1:(length(polygon)-1) # Find a way to pass island vector (vec)
                            intersection_check = intersection(sq_new[1],last(sq_new),polygon[i],polygon[i+1])
                            if intersection_check == true
                                deleteat!(sq_new,length(sq_new))
                                #global loop_ended
                                loop_ended = false
                                break
                            end
                        end
                
                        if loop_ended == true
                            push!(sq_new,sq_new[1])
                        end
                
                    end
                end
                
                return sq_new
            
            end

            return add_last_point(remove_consecutive_collinears(sort_points_inter(sq,polygon)))

        end



        for i in 1:length(file_list)
            sq_sorted = sort_points(sq[i])
            sq_sorted_deleted = sorting(sq_sorted,pol[i])
            push!(boundaries,sq_sorted_deleted)
            global loop_ended
            loop_ended = false
        end

        return boundaries
    end

    function manual_boundaries(all_sq,all_pol)

        function sort_and_delete(sq,pol)

            function ask_and_delete(reference_vec)
                k = "Y"
                if k == "Y" || k == "N"
                    while k == "Y"
                        println("Point to be removed (x-coordinate): ")
                        x_coord = parse(Float64,readline())
                        println("Point to be removed (y-coordinate): ")
                        y_coord = parse(Float64,readline())
                        if in((x_coord,y_coord),reference_vec)
                            deleteat!(reference_vec, findall(x->x==(x_coord,y_coord),reference_vec))
                            println("\nDo you want to delete another point? [Y/N]: ")
                            k = readline()
                        else
                            println("The point you enter does not exist.")
                        end
                    end
                else 
                    println("You failed this city.")
                end
                return reference_vec
            
            end
        
            sq = sort_points(sq)
            display(check_plot(sq,pol))
            command = "Y"
            if command == "Y" || command == "N"
                while command == "Y"
                    sq = ask_and_delete(sq)
                    sq = sort_points(sq)
                    display(check_plot(sq,pol))
                    println("Would you like to delete more? [Y/N]: ")
                    command = readline()
                end
            else 
                println("You failed this city.")
            end
        
            return sq
        end

        global file_list
        final_boundaries = Vector{Tuple{Float64,Float64}}[]

        for i in 1:length(all_pol)
            println(file_list[i])
            sorted = sort_points(all_sq[i])
            display(check_plot(sorted,all_pol[i]))
            println("Check the printed plot. Would you like to remove points? [Y/N]: ")
            decision = readline()
            if decision == "Y"
                new_boundary = []
                new_boundary = sort_and_delete(sorted,all_pol[i])
                push!(new_boundary,new_boundary[1])
                display(check_plot(new_boundary,all_pol[i]))
                push!(final_boundaries,new_boundary)
            elseif decision =="N"
                push!(final_boundaries,sorted)
            else
                println("You have failed this city.")
            end
        end
        return final_boundaries
    end

    function load_files(folder_name)

        boundaries = Vector{Tuple{Float64,Float64}}[]

        for file in readdir(joinpath(pwd(),folder_name))
            boundary_data = readdlm(joinpath(pwd(),folder_name,file))

            boundary = Tuple{Float64,Float64}[]
            for i in 1:length(boundary_data[1:end,1])
                push!(boundary,(boundary_data[i,1], boundary_data[i,2]))
            end
            push!(boundaries,boundary)
        end

        return boundaries
    end

    function save_to_txt(final_boundaries,folder_name)
        folder = string(pwd(),"\\",folder_name)
        global file_list
        if isdir(folder) == false
            mkdir(folder)
        end


        foreach(rm,readdir(string(pwd(),"\\",folder_name,"\\"),join=true))

        for i in 1:length(file_list)
            file = string(folder,"\\" , file_list[i])

            open(file, "w") do io
                writedlm(io,final_boundaries[i])
            end
        end
    end

    function check_plot(boundary, polygon,nx,ny)
        p = plot(polygon, label = "Polygon", color=:black, dpi=:500, legend=false, grid=true, gridalpha = 0.5,xticks = (0:(1/nx):1), yticks = (0:(1/ny):1))
        p = plot!(p,boundary, label = "Square Boundary", color=:blue)
        p = scatter!(p,boundary, markershape=:circle, markercolor=:red, label = "Points")
        pt = boundary
        for i in 1:length(pt)
            valstr = string(pt[i])
            p = annotate!((pt[i]...,text(valstr, 5, :bottom, :left, :green)))
        end
        return p
    end

    function overall_plot(boundaries,polygons,nx,ny)
        p = plot(polygons[1], label = "Polygon", color=:black, dpi=:500,xlims=(-(1/nx),(1+(1/nx))),ylims=(-(1/ny),(1+(1/ny))),grid=true, gridalpha = 0.5,xticks = ((-(1/nx):(1/nx):(1+(1/nx))),""), yticks =((-(1/ny):(1/ny):(1+(1/ny))),""),legend=:bottomleft)
        p = plot!(p,boundaries[1], label = "Square Boundary", color=:blue)
        p = scatter!(p,boundaries[1], markersize=2, markershape=:circle, markercolor=:red, label = "Points")
        for i in 2:length(file_list)
            p = plot!(p,polygons[i], color=:black,label=nothing)
            p = plot!(p,boundaries[i],color=:blue,label=nothing)
            p = scatter!(p,boundaries[i], markersize=2, markershape=:circle, markercolor=:red,label=nothing)
        end
        return p
    end

    function GoalPointToCartesianIndex(point,nx,ny)

        function optimums(filelist) #Find max and mins to use in VectorOfPolygons/Squares
    
            polyg_data1 = readdlm(joinpath(pwd(),"boundaries",filelist[1]))
            max1 = maximum(polyg_data1[1:end,1])
            min1 = minimum(polyg_data1[1:end,1])
            max2 = maximum(polyg_data1[1:end,2])
            min2 = minimum(polyg_data1[1:end,2])
        
            for file in filelist[2:end]
                polyg_data = readdlm(joinpath(pwd(),"boundaries",file))
                p1_max = maximum(polyg_data[1:end,1])
                p1_min = minimum(polyg_data[1:end,1])
                p2_max = maximum(polyg_data[1:end,2])
                p2_min = minimum(polyg_data[1:end,2])
        
                if p1_max > max1
                    max1 = p1_max
                end
        
                if p1_min < min1
                    min1 = p1_min
                end
        
                if p2_max > max2
                    max2 = p2_max
                end
        
                if p2_min < min2
                    min2 = p2_min
                end
            end
            return max1, min1, max2, min2
        end
    
        max_x = optimums(file_list)[1]
        min_x = optimums(file_list)[2]
        max_y = optimums(file_list)[3]
        min_y = optimums(file_list)[4]
    
        lmbda = (max_x-min_x)/(max_y-min_y)
    
        normalised_x = lmbda*(point[1]-min_x)/(max_x-min_x)+0.25
        normalised_y = (point[2]-min_y)/(max_y-min_y)
    
        cartesian_x = convert(Int64,round(normalised_x*nx,RoundUp))
        cartesian_y = convert(Int64,round(normalised_y*ny,RoundUp))
    
        #return (normalised_x,normalised_y)
        return CartesianIndex(cartesian_x,cartesian_y)
    end

    #-----Check if point inpolygon | 0: out of polygon 1: inside polygon -1: on polygon
    function inanypolygon(point, polygons)
        for poly in polygons
            if abs(inpolygon(point,poly)) == 1
                return true
            end
        end
        return false
    end

    #-----Transformations
    function TuplesToVectorOfFloats(vector_of_squares)

        output = Vector{Float64}[]
        for i in 1:length(vector_of_squares)
            val1 = vector_of_squares[i][1]
            val2 = vector_of_squares[i][2]
            push!(output,[val1, val2])
        end

        return output

    end

    function VectorOfFloatsToTuples(vector_test)

        output = Tuple{Float64,Float64}[]
        for i in 1:length(vector_test)
            val1 = vector_test[i][1]
            val2 = vector_test[i][2]
            push!(output,(val1,val2))
        end

        return output

    end

end
