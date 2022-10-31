#Import Packages
begin
    using DataFrames
    using CSV
    using Plots
    using Statistics
    using DelimitedFiles
    using StatsBase
    using Flux
    using Metrics
    using MLDataUtils
    using StatsBase
    using Dates
    using Flux.Data
    using Hyperopt
    using BSON: @save, @load
end

#Read Data
begin 
    #----------Read file
    file = joinpath(pwd(),"unified_data.csv");
    df = CSV.read(file,DataFrame;ignoreemptyrows=false);

    #----------Data
    timestamp = df[!,"Parameters.parameter_timestamp"]
    #-------------------Fuel
    hfo_usage = df[!,"meonhfo"]

    #-------------------Hull and Propeller Cleaning
    hullcleaning = df[!,"time_hullcleaning"]
    properllerpolish = df[!,"time_propellerpolishing"]


    #-------------------Voyage
    voyage_code = df[!,"voyagecode"]

    #--------------------Speed
    #------------------------------Plan on using
    speed_tw = df[!,"stw"]
    speed_instr = df[!,"instructedspeed"]

    #------------------------------Probably won't use
    speed_shaft = df[!,"shaftspeed"]
    speed_og = df[!,"sog"]


    #--------------------Fuel Oil Supplies
    #------------------------------Plan on using
    mefo_supplyin = df[!,"mefomass_supplyin"] #mt/day


    #------------------------------Probably won't use
    me_sfoc = df[!,"mesfoc"]; #not ISO corrected
    me_load = df[!,"meloadperc"];
    me_shp = df[!,"meshp"];
    mefo_density = df[!,"mefosupden"];

    #--------------------Draught
    #------------------------------Plan on using
    t_aft = df[!,"taft"];
    t_fwd = df[!,"tfwd"];
    t_mid_port = df[!,"tmidp"];
    t_mid_star = df[!,"tmids"];

    #------------------------------Probably won't use
    bottom_depth = df[!,"underkeel"];
    r_taft = df[!,"r_draftaft"];
    r_tfwd = df[!,"r_draftfwd"]; 

    #--------------------Trim
    #------------------------------Plan on using
    trim = df[!,"triminclinometer"];
    tranv_trim = df[!,"listinclinometer"];

    #------------------------------Probably won't use
    c_tranv_trim = df[!,"c_listinclinometer"];
    c_trim_deg = df[!,"c_triminclinometer"];
    c_trim_m = df[!,"c_triminclinometer_m"];

    #--------------------Weather
    #------------------------------Plan on using
    wind_dir_relative = df[!,"winddirrel"];
    wind_speed_relative = df[!,"windspeedrel"];

    surface_current_speed = df[!,"WeatherData.sea_surface_current_speed"];
    surface_current_direction = df[!,"WeatherData.sea_surface_current_direction"];


    sign_waves_height = df[!,"WeatherData.significant_waves_height"]; #m
    sign_waves_direction = df[!,"WeatherData.significant_waves_direction"]; #deg
    mean_wave_period = df[!,"WeatherData.mean_wave_period"]; #sec

    #------------------------------Probably won't use
    heading = df[!,"heading"]; #compass [deg] 
    wind_speed_abs = df[!,"windspeedabs"];
    wind_direction = df[!,"WeatherData.wind_direction"]; #deg
    wind_speed = df[!,"WeatherData.wind_speed"]; #knots
    wind_waves_height = df[!,"WeatherData.wind_waves_height"]; #m
    wind_waves_direction = df[!,"WeatherData.wind_waves_direction"]; #deg

    #--------------------Position & Distance
    #------------------------------Plan on using
    distance_tw = df[!,"distancetw"]; #nm
    distance_og = df[!,"distanceog"]; #nm

    #------------------------------Probably won't use
    latitude = df[!,"latitude"];
    longitude = df[!,"longitude"];

    #--------------------Rudder Angle
    #------------------------------Plan on using
    rate_of_turn = df[!,"rotav"];
    rudder_angle_rms = df[!,"rudderanglerms"];
end

#Functions used
begin 

    function dateTimestamp(date)

        #Function dateTimestamp explained: Takes as input one date from raw data and transforms it to 
        #an object from package Dates in order to check later the time distance of two data points
        function readTime(date)
    
            if string(date[2]) == "/" #month = 1 digit
                month = parse(Int64,date[1])
                if string(date[4]) == "/" #day = 1 digit
                    day = parse(Int64,date[3])
                    year = 2000 + parse(Int64,date[5:6])
                    if string(date[10]) == ":"
                        hour = parse(Int64,date[9])
                        minutes = parse(Int64,date[11:end-3])
                    elseif string(date[11]) == ":"
                        hour = parse(Int64,date[9:10])
                        minutes = parse(Int64,date[12:end-3])
                    end
                elseif string(date[5]) == "/" #day = 2 digits 
                    day = parse(Int64,date[3:4])
                    year = 2000 + parse(Int64,date[6:7])
                    if string(date[11]) == ":"
                        hour = parse(Int64,date[10])
                        minutes = parse(Int64,date[12:end-3])
                    elseif string(date[12]) == ":"
                        hour = parse(Int64,date[10:11])
                        minutes = parse(Int64,date[13:end-3])
                    end
                end
            elseif string(date[3]) == "/" #month = 2 digits
                month = parse(Int64,date[1:2])
                if string(date[5]) == "/" #day = 1 digit
                    day = parse(Int64,date[4])
                    year = 2000 + parse(Int64,date[6:7])
                    if string(date[11]) == ":"
                        hour = parse(Int64,date[10])
                        minutes = parse(Int64,date[12:end-3])
                    elseif string(date[12]) == ":"
                        hour = parse(Int64,date[10:11])
                        minutes = parse(Int64,date[13:end-3])
                    end
                elseif string(date[6]) == "/" #day = 2 digits 
                    day = parse(Int64,date[4:5])
                    year = 2000 + parse(Int64,date[7:8])
                    if string(date[12]) == ":"
                        hour = parse(Int64,date[11])
                        minutes = parse(Int64,date[13:end-3])
                    elseif string(date[13]) == ":"
                        hour = parse(Int64,date[11:12])
                        minutes = parse(Int64,date[14:end-3])
                    end
                end
            end
    
            if string(date[end-1:end]) == "PM" && hour < 12
                add12 = 12
            elseif string(date[end-1:end]) == "AM"
                add12 = 0
            elseif string(date[end-1:end]) == "PM" && hour == 12
                add12 = 0
            end
        
        
            return day, month, year, add12 + hour, minutes
        end
    
        day,month,year,hour,minutes = readTime(date)
        
        return Dates.DateTime(year,month,day,hour,minutes)
    
    end

    function training_results(model,Xtrain,Ytrain,Xtest,Ytest)
        trainData_prediction = model(Xtrain)'
        testData_prediction = model(Xtest)'
    
        #R2 Scores
        train_accuracy = r2_score(Ytrain,trainData_prediction)
        test_accuracy = r2_score(Ytest,testData_prediction)
    
        #Mean Square Error
        train_MSE = mse(Ytrain,trainData_prediction)
        test_MSE = mse(Ytest,testData_prediction)
    
        #Mean Square Error
        train_MAE = mae(Ytrain,trainData_prediction)
        test_MAE = mae(Ytest,testData_prediction)
    
        #Printing Results
        print("Train Comb. Accuracy: ", round(train_accuracy;digits=3),"\n")
        print("Test Comb. Accuracy: ", round(test_accuracy;digits=3),"\n")
    
        print("Mean Comb. Square Error in Train Dataset: ", round(train_MSE;digits=6),"\n")
        print("Mean Comb. Square Error in Test Dataset: ", round(test_MSE;digits=6),"\n")
    
        print("Mean Comb. Absolute Error in Train Dataset: ", round(train_MAE;digits=6),"\n")
        print("Mean Comb. Absolute Error in Test Dataset: ", round(test_MAE;digits=6),"\n")
    
    end
    
    function print_correlations(data,data_names)
        print("Correlation Data")
        print("\n")
        for i in 2:length(data)
            print("--",data_names[i]," =   ")
            correl = round(Statistics.cor(data[1],data[i]);digits = 2)
            print(correl)
            print("\n")
        end
    end
    
    function histograms(data,data_names)
        plot_array = []
        for i in 1:length(data)
            plt = histogram(data[i], color="blue",bins=100,alpha = 0.5, xlabel = data_names[i], ylabel = "Frequency")
            push!(plot_array,plt)
        end
    
        p = plot([p for p in plot_array]...,size = (2000,1000))
        
        return p
    end

    function nnmodel(input,h1,h2,h3,h4,activation)
        l1 = Dense(input,h1,activation)
        l2 = Dense(h1,h2,activation)
        l3 = Dense(h2,h3,activation)
        l4 = Dense(h3,h4,activation)
        l5 = Dense(h4,1)
        return Chain(l1,l2,l3,l4,l5)
    end

end

#Filter Data
# 0. Consrtuct significant waves relative direction 
sign_waves_relative = sign_waves_direction - heading
SWR_rad = passmissing(deg2rad).(sign_waves_relative)
SWHR = sign_waves_height.*passmissing(sin).(SWR_rad)

current_directionRel = surface_current_direction - heading
current_directionRel_rad = passmissing(deg2rad).(current_directionRel)
currentSpeed_rel = surface_current_speed.*passmissing(sin).(current_directionRel_rad)

t_mid = (t_aft + t_fwd)/2

# 1.    The selected data after the feature reduction method
# Always make sure the timestamp to be the ending column !!! You can add an evaluation for this
# Always make sure the voyage code to be the end-1 column !!! You can add an evaluation for this
selected_data = [mefo_supplyin,speed_og,t_aft,t_fwd,distance_og,sign_waves_height,hfo_usage,voyage_code,timestamp]
fcColumn = 1;
speedColumn = 2;
draughtColumn = 3;
distanceColumn = 5;
hfoColumn = 7;
voyageColumn = 8;
timeColumn = 9;

# Comments: It works fine. You can make it perfect if: 
# - You include when choosing steps, to complete missing and for steps smaller than what you chose
# - You include calculating missing values from outside to inside 
# (in order to not have the same values for every calculated missing value)
function completeMissings(data,steps)

    function calculateMissing(batchData)
        #Takes each mini batch and for each element
        insideCounter = 0
        for el_idx in 1:length(batchData)
            # the mean value is calculated, but only for the inner elements since it has been checked
            # that the outer are not missing from the main code
            if isequal(batchData[el_idx], missing)  
                batchData[el_idx] = mean(skipmissing(batchData))
                insideCounter += 1
            end
        
        # What would be best to change here, 
        # is the missing value to be completed in steps from outside to inside
        end
        return batchData,insideCounter
    end
    
    final = []
    counter = 0

    #specificData refers to each vector
    #inside the vector of vectors that is used as input
    for specificData in data

        #i refers to the element of each vector
        for i in 1:(length(specificData)-steps+1)
        
            #Takes all possible mini batches and separates them
            d = []
            for ii in i:(steps+i-1)
                d = vcat(d,specificData[ii])
            end
        
            #d contains vectors with all the mini batches

            #Checks every mini batch. If the first and last value are not missing goes inside 
            if isequal(first(d),missing) == false && isequal(last(d),missing) == false
                # only if the inner elements are missing, they will be calculated and replaced
                # else, nothing will happen
                call = calculateMissing(d)
                d = call[1]
                counter += call[2]
            end
        
            for idxStep in 1:steps
                specificData[i+idxStep-1] = d[idxStep]
            end
        end
        push!(final,specificData)
    end

    print("Number of gained data points:    ",counter,"\n")
    return final
end

# 2.    Data after calculating some missing values
# Comments: If you use steps = 3 or 5, you do not gain anything. You gain 60 data points for steps=7.
selectedData = completeMissings(selected_data,5);

# 3.    Transform distance data to distance per step
function stepDistance(data;distanceColumn,timeColumn)
    nData = deepcopy(data)
    nData = mapreduce(permutedims,vcat,nData)
    nData = permutedims(nData)

    print("Initial number of data points:       ", size(nData)[1],"\n")

    # Delete all missing data points from missing column
    function deleteMissing_fromColumn(data,column)
        i = 1
        while i < size(data)[1]
            if data[i,column] === missing
                data = data[1:end .!= i,:]
                i -= 1
            end
            i += 1
        end
        return data
    end

    nData = deleteMissing_fromColumn(nData,distanceColumn)
    print("Number of data points left:      ", size(nData)[1],"\n")

    # Put all the data in time order
    function timeOrder(fdata,timeColumn)
        #These stamps are needed in order to sort the results back to time order
        stamps = []
        for i in 1:size(fdata)[1]
            push!(stamps,dateTimestamp(fdata[i,timeColumn]))
        end

        # Attaching the time stamps to the matrix
        fdata = hcat(fdata,stamps)

        # Sorting the matrix according to the time stamps
        fdata = fdata[sortperm(fdata[:, end];rev=true), :]

        #Discarding the time stamps
        return fdata = fdata[1:end,1:end .!= end]
    end

    nData = timeOrder(nData,timeColumn)

    # Split to sequential data
    function timeSequentialData(data,timeColumn)

        function time_breakPoints(data,timeposition)
    
            function dateTimestamp(date)
        
                #Function dateTimestamp explained: Takes as input one date from raw data and transforms it to 
                #an object from package Dates in order to check later the time distance of two data points
                function readTime(date)
            
                    if string(date[2]) == "/" #month = 1 digit
                        month = parse(Int64,date[1])
                        if string(date[4]) == "/" #day = 1 digit
                            day = parse(Int64,date[3])
                            year = 2000 + parse(Int64,date[5:6])
                            if string(date[10]) == ":"
                                hour = parse(Int64,date[9])
                                minutes = parse(Int64,date[11:end-3])
                            elseif string(date[11]) == ":"
                                hour = parse(Int64,date[9:10])
                                minutes = parse(Int64,date[12:end-3])
                            end
                        elseif string(date[5]) == "/" #day = 2 digits 
                            day = parse(Int64,date[3:4])
                            year = 2000 + parse(Int64,date[6:7])
                            if string(date[11]) == ":"
                                hour = parse(Int64,date[10])
                                minutes = parse(Int64,date[12:end-3])
                            elseif string(date[12]) == ":"
                                hour = parse(Int64,date[10:11])
                                minutes = parse(Int64,date[13:end-3])
                            end
                        end
                    elseif string(date[3]) == "/" #month = 2 digits
                        month = parse(Int64,date[1:2])
                        if string(date[5]) == "/" #day = 1 digit
                            day = parse(Int64,date[4])
                            year = 2000 + parse(Int64,date[6:7])
                            if string(date[11]) == ":"
                                hour = parse(Int64,date[10])
                                minutes = parse(Int64,date[12:end-3])
                            elseif string(date[12]) == ":"
                                hour = parse(Int64,date[10:11])
                                minutes = parse(Int64,date[13:end-3])
                            end
                        elseif string(date[6]) == "/" #day = 2 digits 
                            day = parse(Int64,date[4:5])
                            year = 2000 + parse(Int64,date[7:8])
                            if string(date[12]) == ":"
                                hour = parse(Int64,date[11])
                                minutes = parse(Int64,date[13:end-3])
                            elseif string(date[13]) == ":"
                                hour = parse(Int64,date[11:12])
                                minutes = parse(Int64,date[14:end-3])
                            end
                        end
                    end
            
                    if string(date[end-1:end]) == "PM" && hour < 12
                        add12 = 12
                    elseif string(date[end-1:end]) == "AM"
                        add12 = 0
                    elseif string(date[end-1:end]) == "PM" && hour == 12
                        add12 = 0
                    elseif string(date[end-1:end]) == "AM" && hour == 12
                        add12 = -12
                    end
                
                
                    return day, month, year, add12 + hour, minutes
                end
            
                day,month,year,hour,minutes = readTime(date)
                
                return Dates.DateTime(year,month,day,hour,minutes)
            
            end
        
            non_continuityTracker = []
            # This for loop checks all time differences in order to find out whether two data points have significant time difference
            for i in 2:size(data)[1]
                duration = Dates.Minute(dateTimestamp(data[i,timeposition])-dateTimestamp(data[i-1,timeposition]))
                if abs(duration) != Dates.Minute(10)
                    push!(non_continuityTracker,i)
                end
            end
            return non_continuityTracker
        end
    
        # We find the break points in the matrix data
        breakPoints = time_breakPoints(data,timeColumn)
    
        # We push the first batch to a vector in order to initialize it
        splitData = []
        push!(splitData,data[1:(first(breakPoints)-1),1:end])
    
        # We push the rest of the sequential data inside the vector
        for i in 2:length(breakPoints)
            push!(splitData,data[breakPoints[i-1]:(breakPoints[i]-1),1:end])
        end
    
        # We push the final batch to the vector
        push!(splitData,data[last(breakPoints):end,1:end])
    
        return splitData
    end

    vecData = timeSequentialData(nData,timeColumn)

    # Subtract the distances for each data batch
    disData = deepcopy(vecData)
    i = 1
    while i<=length(disData)
        if size(disData[i])[1] > 1
            for ii in size(disData[i])[1]-1:-1:1
                disData[i][ii,distanceColumn] = - vecData[i][ii+1,distanceColumn] + vecData[i][ii,distanceColumn]
            end
            disData[i] = disData[i][1:end .!= end, :]
        else 
            disData = disData[1:end .!= i]
            vecData = vecData[1:end .!= i]
            i -= 1 
        end
        i += 1
    end

    disData = mapreduce(permutedims,hcat,disData)
    disData = permutedims(disData)

    return disData

end

distanceData = stepDistance(selectedData;distanceColumn,timeColumn)

# 4.   Delete missing values
# 4.   Get rid of the hfo_usage
function filterData(data;hfoColumn,voyageColumn,speedColumn,draughtColumn,fcColumn)

    function cleardataBelow(data,pos,threshold)
        #This function required the criterion category to be in the first place
        i = 1
        while i < size(data)[1]
            if data[i,pos] < threshold
                data = data[1:end .!= i,:]
                i -=1
            end
            i += 1
        end
        return data
    end

    function cleardataOver(data,pos,threshold)
        #This function required the criterion category to be in the first place
        i = 1
        while i < size(data)[1]
            if data[i,pos] > threshold
                data = data[1:end .!= i,:]
                i -=1
            end
            i += 1
        end
        return data
    end

    #Have all data in the appropriate form
    function deleteMissings(data,excludeColumns)
        editData = deepcopy(data)
        i = 1
        #check every line (row)
        while i <= size(editData)[1]
            #check in every row at each column
            for feature_idx in setdiff(collect(1:size(editData)[2]),excludeColumns)
                #if a missing point is spotted
                if editData[i,feature_idx] === missing
                    #delete for every column at the same spot
                    editData = editData[1:end .!= i,:]
                    i -= 1
                    break
                end
            end
            i += 1
        end
        return editData
    end
    fdata = deleteMissings(data,voyageColumn)
    print("Status:  All missing data got rid.","\n")
    print("Number of data points:   ", size(fdata)[1],"\n")

    #Clear all outlier points of Fuel Consumption
    fdata = cleardataBelow(fdata,fcColumn,0)
    print("Number of data points [After FC Filter]:   ", size(fdata)[1],"\n")
    
    #Clear all outlier SPEED points (Speed<3knots)
    fdata = cleardataBelow(fdata,speedColumn,0)
    print("Number of data points [After Speed Filter]:   ", size(fdata)[1],"\n")
    
    #Clear all outlier DRAUGHT points(Draught<6m)
    fdata = cleardataBelow(fdata,draughtColumn,5)
    fdata = cleardataBelow(fdata,draughtColumn+1,5)
    print("Number of data points [After Draught Filter]:   ", size(fdata)[1],"\n")
    
    #Clear all NON HFO Data
    function clear_nonhfo(data,hfo_column)
        i = 1
        while i < size(data)[1]
            if data[i,hfo_column] != 1 
                data = data[1:end .!= i,1:end]
                i -=1
            end
            i += 1
        end
        return data
    end

    fdata = clear_nonhfo(fdata,hfoColumn)
    print("Number of data points [After HFO Filter]:   ", size(fdata)[1],"\n")

    #Clear all Distance outliers
    #fdata = cleardataOver(fdata,5,3)
    #fdata = cleardataBelow(fdata,5,10^(-5))
    #print("Number of data points [After Distance Filter]:", size(fdata)[2],"\n")

    #Get rid of HFO column
    fdata = fdata[1:end,1:end .!= hfoColumn] 

    return fdata
end

@time filteredSelected_data = filterData(distanceData;hfoColumn,voyageColumn,speedColumn,draughtColumn,fcColumn)

# 5.   Split data to Ballast Condition and Loaded Condition
function draughtSplit(data,threshold;draught_column=draughtColumn)
    i = 1
    k = 1
    departure = Matrix{Any}(undef,0,size(data)[2])
    ballast = Matrix{Any}(undef,0,size(data)[2])
    #check every line (row)
    for i in 1:length(data[1:end,1])
        #check in every row at each column
            if data[i,draught_column] < threshold
                #delete for every column at the same spot
                ballast = vcat(ballast,permutedims(data[i,1:end]))
                i -= 1
            else
                departure = vcat(departure,permutedims(data[i,1:end]))
                i -= 1
            end    
        k += 1
        if mod(k,1000) == 0
            print("Progress Status:    ",round(100*k/length(data[1:end,1]);digits=2),"%","\n")
        end
    end

    return ballast, departure
end

splittedDraught_data = draughtSplit(filteredSelected_data,12.5)
ballast = splittedDraught_data[1]
loaded = splittedDraught_data[2]

# 6. Take care of outliers
function outliersRejection(data,k;interestColumn,detectionColumns,timeColumn,lowerBound,upperBound)

    split = [i for i in lowerBound:0.5:upperBound]
    rangeGroups = []
    counterGroupConstruction = 0

    #Filtering the data for outliers
    for groupIdx in 2:length(split)

        # Insert data into the range groups
        function groupConstruction(data,lowerLimit,upperLimit,interestColumn)
            tempArray = Matrix{Any}(undef,0,size(data)[2])
            for rowIdx in 1:size(data)[1]
                if data[rowIdx,interestColumn] >= lowerLimit && data[rowIdx,interestColumn] < upperLimit
                    tempArray = vcat(tempArray,permutedims(data[rowIdx,:]))
                end
            end
            tempArray
        end

        tempArray = groupConstruction(data,split[groupIdx-1],split[groupIdx],interestColumn)
        counterGroupConstruction += size(tempArray)[1]

        if !isempty(tempArray)

            # Calculate the mean values and variance for each category inside the group
            function calculateMeanVariance(data)
                meanValues = []
                varianceValues = []
        
                for columnIdx in 1:size(data)[2]-1
                    push!(meanValues,mean(data[:,columnIdx]))
                    push!(varianceValues,Statistics.std(data[:,columnIdx]))
                end
                return meanValues,varianceValues
            end

            meanValues, varianceValues = calculateMeanVariance(tempArray)

            # Delete all data that are considered outliers
            function deleteOutliers(data,categoryIdxs,meanValues,varianceValues,k)
                for categoryIdx in categoryIdxs
                    poutsa1 = 0
                    rowIdx = 1
                    while rowIdx <= size(data)[1]
                        if abs(data[rowIdx,categoryIdx] - meanValues[categoryIdx])/varianceValues[categoryIdx] > k
                            data = data[1:end .!= rowIdx,1:end]
                            rowIdx -= 1
                            poutsa1 += 1
                        end
                        rowIdx += 1
                    end
                end
        
                return data
            end

            tempArray = deleteOutliers(tempArray,detectionColumns,meanValues,varianceValues,k)
            rangeGroups = push!(rangeGroups,tempArray)

        end

    end

    # Convert the vector that you produced, back to the matrix form
    finalArray = first(rangeGroups)
    for range in rangeGroups[2:end]
        finalArray = vcat(finalArray,range)
    end

    print("Number of data points deleted:   ", size(data)[1]-size(finalArray)[1],"\n")
    #print("counterGroupConstruction:   ", counterGroupConstruction,"\n")

    #These stamps are needed in order to sort the results back to time order
    stamps = []
    for i in 1:size(finalArray)[1]
        push!(stamps,dateTimestamp(finalArray[i,timeColumn]))
    end

    # Attaching the time stamps to the matrix
    finalArray = hcat(finalArray,stamps)

    # Sorting the matrix according to the time stamps
    finalArray = finalArray[sortperm(finalArray[:, timeColumn+1];rev=true), :]

    #Discarding the time stamps
    finalArray = finalArray[1:end,1:end .!= end]

    return finalArray
end

ballast_noOutliers = outliersRejection(ballast,3.5; interestColumn = speedColumn,detectionColumns = [fcColumn; distanceColumn],timeColumn = timeColumn-1,lowerBound = 0,upperBound = 16)
loaded_noOutliers = outliersRejection(loaded,3.5; interestColumn = speedColumn,detectionColumns = [fcColumn; distanceColumn],timeColumn = timeColumn-1,lowerBound =0 ,upperBound =16)
total_noOutliers = outliersRejection(filteredSelected_data,3.5; interestColumn = speedColumn,detectionColumns = [fcColumn; distanceColumn],timeColumn = timeColumn-1,lowerBound=0, upperBound =16)

# 7. ZScoreTransformation 
function ZScoreTransformation(data)
    # We transform the data to ZScoreTransform
    xTransform = StatsBase.fit(ZScoreTransform,convert(Matrix{Float32},data[:,2:(end-2)])')
    yTransform = StatsBase.fit(ZScoreTransform,convert(Vector{Float32},data[:,1]))

    x = permutedims(StatsBase.transform(xTransform,convert(Matrix{Float32},data[:,2:(end-2)])'))
    y = StatsBase.transform(yTransform,convert(Vector{Float32},data[:,1]))

    # We unify our data, joining the time stamps
    newData = hcat(y,x,data[:,end-1:end])

    return newData, xTransform, yTransform
end

ballastTransformation = ZScoreTransformation(ballast_noOutliers)
ballastZ = ballastTransformation[1]
ballastZ_x = ballastTransformation[2]
ballastZ_y = ballastTransformation[3]

loadedTransformation = ZScoreTransformation(loaded_noOutliers);
loadedZ = loadedTransformation[1]
loadedZ_x = loadedTransformation[2]
loadedZ_y = loadedTransformation[3]

totalTransformation = ZScoreTransformation(total_noOutliers);
totalZ = totalTransformation[1]
totalZ_x = totalTransformation[2]
totalZ_y = totalTransformation[3]

open("inputsMeanDeviation.txt", "w") do io
    writedlm(io,[totalZ_x.mean totalZ_x.scale])
end

open("labelMeanDeviation.txt", "w") do io
    writedlm(io,[totalZ_y.mean totalZ_y.scale])
end

# 8. Sequential Data Construction
function timeSequentialData(data;timeColumn=timeColumn-1)

    function time_breakPoints(data,timeposition)

        function dateTimestamp(date)
    
            #Function dateTimestamp explained: Takes as input one date from raw data and transforms it to 
            #an object from package Dates in order to check later the time distance of two data points
            function readTime(date)
        
                if string(date[2]) == "/" #month = 1 digit
                    month = parse(Int64,date[1])
                    if string(date[4]) == "/" #day = 1 digit
                        day = parse(Int64,date[3])
                        year = 2000 + parse(Int64,date[5:6])
                        if string(date[10]) == ":"
                            hour = parse(Int64,date[9])
                            minutes = parse(Int64,date[11:end-3])
                        elseif string(date[11]) == ":"
                            hour = parse(Int64,date[9:10])
                            minutes = parse(Int64,date[12:end-3])
                        end
                    elseif string(date[5]) == "/" #day = 2 digits 
                        day = parse(Int64,date[3:4])
                        year = 2000 + parse(Int64,date[6:7])
                        if string(date[11]) == ":"
                            hour = parse(Int64,date[10])
                            minutes = parse(Int64,date[12:end-3])
                        elseif string(date[12]) == ":"
                            hour = parse(Int64,date[10:11])
                            minutes = parse(Int64,date[13:end-3])
                        end
                    end
                elseif string(date[3]) == "/" #month = 2 digits
                    month = parse(Int64,date[1:2])
                    if string(date[5]) == "/" #day = 1 digit
                        day = parse(Int64,date[4])
                        year = 2000 + parse(Int64,date[6:7])
                        if string(date[11]) == ":"
                            hour = parse(Int64,date[10])
                            minutes = parse(Int64,date[12:end-3])
                        elseif string(date[12]) == ":"
                            hour = parse(Int64,date[10:11])
                            minutes = parse(Int64,date[13:end-3])
                        end
                    elseif string(date[6]) == "/" #day = 2 digits 
                        day = parse(Int64,date[4:5])
                        year = 2000 + parse(Int64,date[7:8])
                        if string(date[12]) == ":"
                            hour = parse(Int64,date[11])
                            minutes = parse(Int64,date[13:end-3])
                        elseif string(date[13]) == ":"
                            hour = parse(Int64,date[11:12])
                            minutes = parse(Int64,date[14:end-3])
                        end
                    end
                end
        
                if string(date[end-1:end]) == "PM" && hour < 12
                    add12 = 12
                elseif string(date[end-1:end]) == "AM"
                    add12 = 0
                elseif string(date[end-1:end]) == "PM" && hour == 12
                    add12 = 0
                elseif string(date[end-1:end]) == "AM" && hour == 12
                    add12 = -12
                end
            
            
                return day, month, year, add12 + hour, minutes
            end
        
            day,month,year,hour,minutes = readTime(date)
            
            return Dates.DateTime(year,month,day,hour,minutes)
        
        end
    
        non_continuityTracker = []
        # This for loop checks all time differences in order to find out whether two data points have significant time difference
        for i in 2:size(data)[1]
            duration = Dates.Minute(dateTimestamp(data[i,timeposition])-dateTimestamp(data[i-1,timeposition]))
            if abs(duration) != Dates.Minute(10)
                push!(non_continuityTracker,i)
            end
        end
        return non_continuityTracker
    end

    # We find the break points in the matrix data
    breakPoints = time_breakPoints(data,timeColumn)

    # We push the first batch to a vector in order to initialize it
    splitData = []
    push!(splitData,data[1:(first(breakPoints)-1),1:end])

    # We push the rest of the sequential data inside the vector
    for i in 2:length(breakPoints)
        push!(splitData,data[breakPoints[i-1]:(breakPoints[i]-1),1:end])
    end

    # We push the final batch to the vector
    push!(splitData,data[last(breakPoints):end,1:end])

    return splitData
end

sequentialBallastData = timeSequentialData(ballastZ);
sequentialLoadedData = timeSequentialData(loadedZ);
sequentialTotalData = timeSequentialData(totalZ);

# 9.   Moving Average Implementation
# For every vector with length > n, the moving average is applied. 
# All vectos with length > n, will lose n-1 elements
function movingData(timeVector,n)

    function moving_average(vector,n)
        newVector = []

        #initialize newVector with the first values
        movingAverage_value = mean(vector[1:n])
        push!(newVector,movingAverage_value)
    
        for ii in (n+1):length(vector)
            movingAverage_value = mean(vector[ii-n:ii])
            push!(newVector,movingAverage_value)
        end
    
        return newVector
    end

    #Take as input timeVector
    moving = []
    movingTemp = []
    
    for i in 1:length(timeVector)
        if size(timeVector[i])[1] < n
            push!(moving,timeVector[i])
        else
            for ii in 1:size(timeVector[i])[2]-1
                push!(movingTemp,moving_average(timeVector[i][:,ii],n))
            end
            push!(movingTemp,timeVector[i][n:end,end])
            push!(moving,permutedims(mapreduce(permutedims,vcat,movingTemp)))
            movingTemp = []
        end
    end

    return moving

end

sequentialMovingBallastData = movingData(sequentialBallastData,3)
sequentialMovingLoadedData = movingData(sequentialLoadedData,3)
sequentialMovingTotalData = movingData(sequentialTotalData,3)

# 10. Only the sequences with length greater than sequenceLength will be chosen. 
# The rest will be discarded. Also, the timestamp is being discarded. 
function dataLSTM(sequentialData,sequenceLength)

    #Creates all the consecutive data points
    observations = []
    labels = []

    #All the vectors with length<sequenceLength are discarded
    discardedPoints = 0
    for timeBatch_idx in 1:length(sequentialData)
        if length(sequentialData[timeBatch_idx])>=sequenceLength
            for i in 1:(size(sequentialData[timeBatch_idx])[1]-sequenceLength)
                #We also get rid of the timestamp
                push!(observations,sequentialData[timeBatch_idx][i:(i+sequenceLength-1),2:(end-2)]')
                push!(labels,sequentialData[timeBatch_idx][i:(i+sequenceLength-1),1])
            end
        else
            discardedPoints += length(sequentialData[timeBatch_idx])
        end
    end

    X = convert(AbstractVector{Matrix{Float32}},observations)
    Y = convert(AbstractVector{Vector{Float32}},labels)

    function tabularForm(X)
        #Takes as input the vectors and creates the appropriate tabular form
        tabular = Array{Float32,3}(undef,sequenceLength,size(X[1])[1],length(X))
        for group_idx in 1:length(X)
            for feature_idx in 1:size(X[group_idx])[1]
                for timestep in 1:sequenceLength
                    tabular[timestep,feature_idx,group_idx] = X[group_idx][feature_idx,timestep]
                end
            end
        end
        return tabular
    end

    function tabularOutput(Y)
        tabular = Array{Float32,3}(undef,sequenceLength,1,length(Y))
        for group_idx in 1:length(Y)
            for timestep in 1:sequenceLength
                tabular[timestep,1,group_idx] = Y[group_idx][timestep]
            end
        end
    
        return tabular
    end

    tabularX = tabularForm(X)
    tabularY = tabularOutput(Y)

    #print("Number of discarded data points:     ", discardedPoints,"\n")

    return tabularX, tabularY
end

#---Parameters 
sequenceLength = 3; 
splitPoint = 0.7;
batchSize = 12;
lstmNodes = 65;
epsilonOptim =  10^(-4);
epochsNumber = 1000;

#---Data
# Comment: For sequenceLength<8, you do not have any discarded points
X,Y = dataLSTM(sequentialMovingTotalData,sequenceLength);

xTrain = X[:,:,1:Int(round(splitPoint*(size(X)[3]),RoundDown))]
yTrain = Y[:,:,1:Int(round(splitPoint*(size(X)[3]),RoundDown))]

xTest = X[:,:,Int(round(splitPoint*(size(X)[3]),RoundDown))+1:end]
yTest = Y[:,:,Int(round(splitPoint*(size(X)[3]),RoundDown))+1:end]

trainData = DataLoader((xTrain, yTrain), batchsize=batchSize,shuffle=true,partial=false);
testData = DataLoader((xTest, yTest), batchsize=batchSize,shuffle=true,partial=false);

#---Model
function lstmModel(inputSize,hidden)
    l1 = LSTM(inputSize,hidden)
    l2 = Dense(hidden,1)
    return Chain(l1,l2)
end

m = lstmModel(size(xTrain)[2],lstmNodes)

#---LSTM Inner Weights
#m[1].cell.Wi
#m[1].cell.Wh
#m[1].cell.b

#---Parameters
ps = Flux.params(m);

#---Optimizer
opt = ADAM(epsilonOptim);

#---Model Training
global dataTrans = totalZ_y

function modelTraining(epochs,model,ps,data,opt,sequenceLength)

    function prediction(model,x)
        inputs = [x[t,:,:] for t in 1:sequenceLength]
        model(inputs[1]) #warm-up the model
        result = [model(x) for x in inputs[2:end]][end]
        Flux.reset!(model)
        return result
    end

    for i in 1:epochs 
        Flux.reset!(model)
        print("Epoch:   ",i,"\n")
        sampleError = []
        for (x,y) in data
            #The following three lines calculate the gradients for weights and biases
            gs = gradient(ps) do 
                Flux.huber_loss(prediction(model,x), y[end,:,:])
            end 
            #The following line updates weights and biases: w = w - a*gs(w), b = b - a*gs(b)
            Flux.update!(opt,ps,gs)
            #The error is calculated in order to be printed
            batchError = round(100*abs(mean((y[end,:,:]-prediction(model,x))./y[end,:,:])),RoundDown;digits=2)
            push!(sampleError,batchError)
        end
        #The following part is the callback function
        print("~Error:   ",round(mean(sampleError);digits=2),"%","\n")
        #display(scatter(m[1].cell.Wi))
        #display(plotLSTM(voyageLSTM,model,sequenceLength))
    end
end

modelTraining(epochsNumber,m,ps,trainData,opt,sequenceLength)

#---Training Results
function trainingResults(trainData,testData,m,sequenceLength;aver=first(dataTrans.mean),stand=first(dataTrans.scale))

    predictedTrain = []
    actualValuesTrain = []
    for (x,y) in trainData
        Flux.reset!(m)
        inputs = [x[t,:,:] for t in 1:sequenceLength]
        for i in 1:length(m.(inputs)[end])
            Flux.reset!(m)
            push!(predictedTrain,stand*m.(inputs)[end][i]+aver)
            push!(actualValuesTrain,stand*y[end,1,i]+aver)
        end
    end

    predictedTrain = convert(Vector{Float32},predictedTrain)
    actualValuesTrain = convert(Vector{Float32},actualValuesTrain)    

    predictedTest = []
    actualValuesTest = []
    for (x,y) in testData
        inputs = [x[t,:,:] for t in 1:sequenceLength]
        Flux.reset!(m)
        for i in 1:length(m.(inputs)[end])
            Flux.reset!(m)
            push!(predictedTest,stand*m.(inputs)[end][i]+aver)
            push!(actualValuesTest,stand*y[end,1,i]+aver)
        end
    end

    predictedTest = convert(Vector{Float32},predictedTest)
    actualValuesTest = convert(Vector{Float32},actualValuesTest)  

    #R2 Scores
    train_accuracy = r2_score(predictedTrain,actualValuesTrain)
    test_accuracy = r2_score(predictedTest,actualValuesTest)

    #Mean Square Error
    train_MSE = mse(predictedTrain,actualValuesTrain)
    test_MSE = mse(predictedTest,actualValuesTest)

    #Mean Square Error
    train_MAE = mae(predictedTrain,actualValuesTrain)
    test_MAE = mae(predictedTest,actualValuesTest)

    #Printing Results
    print("---R² Scores","\n")
    print("Train Comb. Accuracy: ", round(train_accuracy;digits=3),"\n")
    print("Test Comb. Accuracy: ", round(test_accuracy;digits=3),"\n")
    
    print("---Mean Square Error","\n")
    print("Mean Comb. Square Error in Train Dataset: ", round(train_MSE;digits=6),"\n")
    print("Mean Comb. Square Error in Test Dataset: ", round(test_MSE;digits=6),"\n")

    print("---Mean Absolute Error","\n")
    print("Mean Comb. Absolute Error in Train Dataset: ", round(train_MAE;digits=6),"\n")
    print("Mean Comb. Absolute Error in Test Dataset: ", round(test_MAE;digits=6),"\n")
    Flux.reset!(m)
end

@load "lstmModel.bson" m

m

#x = first(trainData)[1]
#inputs = [x[t,:,:] for t in 1:sequenceLength]
#Flux.reset!(m)
#m.(inputs)[end]

trainingResults(trainData,testData,m,sequenceLength)

#---Voyage Data and Plot Route Sequence
function plotLSTM(data,model,sequenceLength,voyageCode;aver=first(dataTrans.mean),stand=first(dataTrans.scale))
    predicted = []
    actualValues = []
    for (x,y) in data[1]
        Flux.reset!(model)
        inputs = [x[t,:,:] for t in 1:sequenceLength]
        for i in 1:length(model.(inputs)[end])
            Flux.reset!(model)
            push!(predicted,stand*model.(inputs)[end][i]+aver)
            push!(actualValues,stand*y[end,1,i]+aver)
        end
    end

    #varianceData = 100*((predicted - actualValues)./actualValues)
    errorData = predicted - actualValues
    breakPoints = data[2]

    p = plot((1:length(predicted)),actualValues,color="red",label="Y")
    p = plot!((1:length(predicted)),predicted, color="blue",label="model(X)")
    p = scatter!(breakPoints,-ones(length(breakPoints)),color="yellow",markershape=:star5)

    #s1 = scatter((1:length(predicted)),varianceData,label=nothing,color="green",grid=true,ga=0.5,markersize=2,ylims=(-200,200))
    s2 = scatter((1:length(predicted)),errorData,label=nothing,color="orange",grid=true,ga=0.5,markersize=2)



    return plot(
        plot(p,ylabel="Fuel Consumption [t/24h]",legend=:topright,legendfontsize=5,grid=true,ga=0.5),
        #scatter(s1,title="Error [%]"),
        scatter(s2,xlabel="Time Step",ylabel="Error [t/24h]"),
        layout = grid(2, 1, heights=[0.6,0.4]),
        dpi=500,
        titlefont=(11,"Computer Modern"),
        guidefont="Computer Modern",
        guidefontsize = 8,
        plot_title = string("Fuel Consumption Prediction - Voyage Code: ",voyageCode)
    )
end

function voyageData(data,voyageNumber;voyageColumnN,timeColumn = timeColumn-1,n=3,sequenceLength=sequenceLength,batchsize=batchSize)

    # Info: The dataLSTM is modified here. 

    function deleteMissingsRoute(data,voyageColumn)
        editData = deepcopy(data)
        i = 1
        #check every line (row)
        while i <= size(editData)[1]
                #if a missing point is spotted
            if editData[i,voyageColumn] === missing
                #delete for every column at the same spot
                editData = editData[1:end .!= i,:]
                i -= 1
            end
            i += 1
        end
        return editData
    end

    initialVoyage = deleteMissingsRoute(data,voyageColumnN)

    function useVoyage_data(data,voyagenumber,voyageColumn)

        function selectVoyageData(voyage_data,voyage_number)
    
            voyages = unique(voyage_data)
            starting_point = 0
            ending_point = 0 
    
            for i in 1:length(voyage_data)
                if voyage_data[i] == voyages[voyage_number]
                    starting_point = i 
                    break 
                end
            end
    
            for ii in length(voyage_data):-1:1
                if voyage_data[ii] == voyages[voyage_number]
                    ending_point = ii 
                    break 
                end
            end
            return starting_point,ending_point
        end
    
        #Find out voyage boundaries data
        bounds = selectVoyageData(data[1:end,voyageColumn],voyagenumber)
       
        function keep_voyage_data(vdata, starting_point, ending_point)
            vdata = vdata[starting_point:ending_point,1:(size(vdata)[2])]
        end
    
        return keep_voyage_data(data,bounds[1],bounds[2])
    
    end

    voyage = useVoyage_data(initialVoyage,voyageNumber,voyageColumnN)

    voyageSeq = timeSequentialData(voyage;timeColumn)

    voyageMovSeq = movingData(voyageSeq,n)

    function dataLSTM(sequentialData,sequenceLength)

        #Creates all the consecutive data points
        observations = []
        labels = []
    
        continuityCounter = 0
        continuityArchive = []

        #All the vectors with length<sequenceLength are discarded
        discardedPoints = 0
        for timeBatch_idx in 1:length(sequentialData)
            if length(sequentialData[timeBatch_idx])>=sequenceLength
                for i in 1:(size(sequentialData[timeBatch_idx])[1]-sequenceLength)
                    #We also get rid of the timestamp
                    push!(observations,sequentialData[timeBatch_idx][i:(i+sequenceLength-1),2:(end-2)]')
                    push!(labels,sequentialData[timeBatch_idx][i:(i+sequenceLength-1),1])
                    continuityCounter +=1 
                end
                continuityArchive = push!(continuityArchive,continuityCounter)
            else
                discardedPoints += length(sequentialData[timeBatch_idx])
            end
        end
    
        X = convert(AbstractVector{Matrix{Float32}},observations)
        Y = convert(AbstractVector{Vector{Float32}},labels)
    
        function tabularForm(X)
            #Takes as input the vectors and creates the appropriate tabular form
            tabular = Array{Float32,3}(undef,sequenceLength,size(X[1])[1],length(X))
            for group_idx in 1:length(X)
                for feature_idx in 1:size(X[group_idx])[1]
                    for timestep in 1:sequenceLength
                        tabular[timestep,feature_idx,group_idx] = X[group_idx][feature_idx,timestep]
                    end
                end
            end
            return tabular
        end
    
        function tabularOutput(Y)
            tabular = Array{Float32,3}(undef,sequenceLength,1,length(Y))
            for group_idx in 1:length(Y)
                for timestep in 1:sequenceLength
                    tabular[timestep,1,group_idx] = Y[group_idx][timestep]
                end
            end
        
            return tabular
        end
    
        tabularX = tabularForm(X)
        tabularY = tabularOutput(Y)
    
        #print("Number of discarded data points:     ", discardedPoints,"\n")
    
        return tabularX, tabularY, unique(continuityArchive)
    end

    voyageNetworkX,voyageNetworkY,continuityArch = dataLSTM(voyageMovSeq,sequenceLength)

    voyagedData = DataLoader((voyageNetworkX, voyageNetworkY), batchsize=batchsize,partial=false)

    return voyagedData,continuityArch

end

voyageCode = 1;

voyageLSTM = voyageData(totalZ,voyageCode; voyageColumnN = voyageColumn-1)

plotLSTM(voyageLSTM,m,sequenceLength,voyageCode)

for i in 1:8
    voyageCode = i;
    voyageLSTM = voyageData(totalZ,voyageCode; voyageColumnN = voyageColumn-1)
    display(plotLSTM(voyageLSTM,m,sequenceLength,voyageCode))
end

#---Save Model 
@save "lstmModel.bson" m

@load "lstmModel.bson" m
#---Hyperopt
function f(nodes,batchSize,sequenceLength,epsilon,epochsNumber)

    function modelTraining(epochs,model,ps,data,opt,sequenceLength)

        function prediction(x)
            inputs = [x[t,:,:] for t in 1:sequenceLength]
            model(inputs[1]) #warm-up the model
            result = [model(x) for x in inputs[2:end]][end]
            Flux.reset!(model)
            return result
        end

        Flux.reset!(model)
        #cb = runall(cb)
        for i in 1:epochs 
            for (x,y) in data
                gs = gradient(ps) do 
                    Flux.mse(prediction(x), y[end,:,:])
                end
                Flux.update!(opt,ps,gs)
            end
        end
    end

    X,Y = dataLSTM(sequentialMovingTotalData,sequenceLength)

    xTrain = X[:,:,1:Int(round(0.7*(size(X)[3]),RoundDown))]
    yTrain = Y[:,:,1:Int(round(0.7*(size(X)[3]),RoundDown))]

    xTest = X[:,:,Int(round(0.7*(size(X)[3]),RoundDown))+1:end]
    yTest = Y[:,:,Int(round(0.7*(size(X)[3]),RoundDown))+1:end]

    trainData = DataLoader((xTrain, yTrain), batchsize=batchSize,shuffle=true,partial=false)
    testData = DataLoader((xTest, yTest), batchsize=batchSize,shuffle=true,partial=false)

    model = lstmModel(size(xTrain)[2],nodes)
    #---Parameters
    ps = Flux.params(model)
    #---Optimizer
    opt = ADAM(epsilon)

    modelTraining(epochsNumber,model,ps,trainData,opt,sequenceLength)

    #Evaluate
    predictedTest = []
    actualValuesTest = []
    for (x,y) in testData
        inputs = [x[t,:,:] for t in 1:sequenceLength]
        for i in 1:length(model.(inputs)[end])
            Flux.reset!(model)
            push!(predictedTest,model.(inputs)[end][i])
            push!(actualValuesTest,y[end,1,i])
        end
    end
    predictedTest = convert(Vector{Float32},predictedTest)
    actualValuesTest = convert(Vector{Float32},actualValuesTest) 

    mse(predictedTest,actualValuesTest)
end

ho = @hyperopt for i=100,
        sampler = RandomSampler(), # This is default if none provided
        nodes = collect(10:5:100),
        batchSize = collect(4:4:32),
        sequenceLength = collect(2:1:4),
        epsilon = LinRange(10^(-5),10^(-3),30)

    print("\n","Sample Number:","\t", i,"\n")
    print("Nodes:","\t", nodes, "\t","Batch Size:","\t", batchSize,"\t","Sequence Length:", "\t",sequenceLength,"\t","Learning Rate:","\t",round(epsilon;digits=7),"\t")
    @show f(nodes,batchSize,sequenceLength,epsilon,500)
end

printmax(ho)
plot(ho)