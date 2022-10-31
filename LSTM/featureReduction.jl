#Import Packages
begin
    using DataFrames
    using CSV
    using Plots
    using Statistics
    using DelimitedFiles
    using ScikitLearn
    using StatsBase
    using Flux
    using Metrics
    using MLDataUtils
    using Dates
end

@sk_import feature_selection: f_regression
@sk_import feature_selection: mutual_info_regression
@sk_import feature_selection: SelectKBest
@sk_import decomposition: PCA

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

# 1.    The selected data after the feature reduction method
# Always make sure the timestamp to be the ending column !!! You can add an evaluation for this
selected_data = [mefo_supplyin,
speed_og,
t_aft, t_fwd, distance_og,
sign_waves_height, mean_wave_period,
wind_speed_relative, 
currentSpeed_rel,
properllerpolish, hullcleaning, 
hfo_usage, 
timestamp]

fdata_names = ["FC",
"SOG",
"T AFT","T FWD","Dist OG",
"SGW","MWP",
"WS Rel",
"Cur S Rel",
"Prop", "Hull"]

fcColumn = 1;
speedColumn = 2;
draughtColumn = 3;
distanceColumn = 5;
hfoColumn = 12;
timeColumn = 13;

# !!! You should maybe fix the calculateMissing
# !!! Add how many missings have been gained. 
# !!! Fix the missing procedure
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
function filterData(data;hfoColumn,speedColumn,draughtColumn,fcColumn)

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

    fdata = deleteMissings(data,[])
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

filteredSelected_data = filterData(distanceData;hfoColumn,speedColumn,draughtColumn,fcColumn)

#Feature Reduction
function featureReductionMetrics(fdata,fdata_names)

    fdata = convert(Matrix{Float64},fdata[:,1:end-1])

    #Z Transformations
    Xtr = StatsBase.fit(ZScoreTransform,fdata[:,2:end])
    Ytr = StatsBase.fit(ZScoreTransform,fdata[:,1])

    Xstd = StatsBase.transform(Xtr,fdata[:,2:end])
    Ystd = StatsBase.transform(Ytr,fdata[:,1])

    #Names
    if length(fdata_names) != size(fdata)[2]
        throw("length(data_names) not equal to length(data)")
    end

    #F-Regression
    F_test = SelectKBest(f_regression,k="all")
    F_test.fit(Xstd,Ystd)

    #Mutual Info Regression
    M_test = SelectKBest(mutual_info_regression,k="all")
    M_test.fit(Xstd,Ystd)

    #Principal Component Analysis
    pca_m = PCA().fit(Xstd)

    #Plot
    return plot(
        bar(fdata_names[2:end],F_test.scores_[1:end],title="F-regression",color="blue",label=nothing,titlefont=(13,"Computer Modern")),
        bar(fdata_names[2:end],M_test.scores_[1:end],title="Mutual Info Regression",color="red",label=nothing,titlefont=(13,"Computer Modern")),
        layout = (2,1)
    ),
    plot(cumsum(pca_m.explained_variance_ratio_),grid=true,gridalpha=.5,xticks=(0:1:size(fdata)[2]),title="Principal Component Analysis",titlefont=(13,"Computer Modern"),label=nothing,xlabel="Number of features",guidefont="Computer Modern", guidefontsize=8,dpi=500)

end

feature_reduction = featureReductionMetrics(filteredSelected_data,fdata_names);
feature_reduction[1] #F-regression, Mutual Info Regression
feature_reduction[2] #Principal Component Analysis

#Calculate Pearson's and Spearman's correlations
function heatmapCorrelation(data;names=fdata_names)

    data = convert(Matrix{Float64},data[:,1:end-1])


    (n,m) = size(cor(data))
    heatmapPearsons = heatmap(cor(data),fc=cgrad([:white,:dodgerblue4]),xticks=(1:m,names),yticks=(1:n,names),xrot=60,yflip=true,title="Pearson's correlation coefficient (r)",titlefont=(15,"Computer Modern"),guidefont="Computer Modern",dpi=500,bottom_margin=9Plots.mm)
    annotate!([(j, i, text(round(cor(data)[i,j],digits=3), 8,"Computer Modern",:black)) for i in 1:n for j in 1:m])

    heatmapSpearman = heatmap(corspearman(data),fc=cgrad([:white,:dodgerblue4]),xticks=(1:m,names),yticks=(1:n,names),xrot=60,yflip=true,title="Spearman's rank correlation coefficient (ρ)",titlefont=(15,"Computer Modern"),guidefont="Computer Modern",dpi=500,bottom_margin=9Plots.mm)
    annotate!([(j, i, text(round(corspearman(data)[i,j],digits=3), 8,"Computer Modern",:black)) for i in 1:n for j in 1:m])

    return heatmapPearsons,heatmapSpearman

end

heatMaps = heatmapCorrelation(filteredSelected_data)
heatMaps[1] #Pearson's Correlation
heatMaps[2] #Spearman's Correlation

plot(filteredSelected_data[:,3])