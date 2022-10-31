#Packages used
begin
    using ReinforcementLearning
    using Flux #Neural Networks functionalities
    using Plots #plotting
    using DelimitedFiles #reading txt files
    using PolygonOps #checks if agent in boundary
    using Random #defining random seed
    using Intervals #defining state_space in normalised process
    using Statistics
    using BSON
    using BSON: @load, @save #for saving the NN parameters
    using Zygote #for loading into the model the saved NN parameters
    using Hyperopt

    include(joinpath(pwd(),"GeoBoundariesManipulation.jl"));
    using .GeoBoundariesManipulation
end

#-------------Load LSTM Model
begin 
    modelPath = string(joinpath(pwd(),"LSTM\\"),"lstmModel.bson")
    lstmFolder = joinpath(pwd(),"LSTM")

    @load modelPath m
    model = deepcopy(m)

    global meanInputs = permutedims(convert(Vector{Float32},readdlm(string(lstmFolder,"\\inputsMeanDeviation.txt"))[:,1]))
    global stdInputs = permutedims(convert(Vector{Float32},readdlm(string(lstmFolder,"\\inputsMeanDeviation.txt"))[:,2]))

    global meanOutput = first(readdlm(string(lstmFolder,"\\labelMeanDeviation.txt"))[:,1])
    global stdOutput = first(readdlm(string(lstmFolder,"\\labelMeanDeviation.txt"))[:,2])
end

#----------Environment Parameters
begin
    struct ShippingEnvParams
        gridworld_dims::Tuple{Int64,Int64} #Gridworld dimensions
        velocities::Vector{Float32} #available velocities from 6 knots to 20 knots
        acceleration::Vector{Float32} #available acceleration per step: -2, 0, 2
        heading::Vector{CartesianIndex{2}} #all heading manoeuvers
        StartingPoint::CartesianIndex{2}
        GoalPoint::CartesianIndex{2}
        all_polygons::Vector{Vector{Tuple{Float64,Float64}}} #all the boundaries
        ETA::Float64 #minutes
    end

    function ShippingEnvParams(;
        gridworld_dims = (16,16),
        velocities = Vector((6.0:2.0:12.0)), #knots
        acceleration = Vector((-2.0:2.0:2.0)), 
        heading = [CartesianIndex(0,1);CartesianIndex(0,-1);CartesianIndex(-1,0);CartesianIndex(-1,1);CartesianIndex(-1,-1);CartesianIndex(1,-1);CartesianIndex(1,1);CartesianIndex(1,0)], 
        StartingPoint = GeoBoundariesManipulation.GoalPointToCartesianIndex((-6.733535,61.997345),gridworld_dims[1],gridworld_dims[2]),
        EndingPoint = GeoBoundariesManipulation.GoalPointToCartesianIndex((-6.691500,61.535580),gridworld_dims[1],gridworld_dims[2]),
        AllPolygons = GeoBoundariesManipulation.load_files("finalboundaries"),
        eta = 10000 #hours
        )
        ShippingEnvParams(
            gridworld_dims,
            velocities,
            acceleration,
            heading,
            StartingPoint,
            EndingPoint,
            AllPolygons,
            eta
        )
    end
end

#-------------Action Space Map
begin
    #--------------------------Parameters: Object Contruction
    struct as_map_params
        nheading::Int64
        nacceleration::Int64
    end
    #--------------------------Define Object values
    function as_map_params(;
        shipping_env_params = ShippingEnvParams(),
        nheading = length(shipping_env_params.heading),
        nacceleration = length(shipping_env_params.acceleration),
        )
        as_map_params(
            nheading,
            nacceleration,
        )
    end
    #--------------------------Final Mapping: Vector of all actions
    function as_map(;map_params = as_map_params())
        
        function remove_internal_tuples(vect)
            d = []
            d_internal = []
            while first(vect)[1] isa Tuple
                d = []
                for i in 1:length(vect)
                    empty!(d_internal)
                    for ii in 1:length(first(vect[i]))
                        push!(d_internal,first(vect[i])[ii])
                    end
                    for iii in 2:length(vect[i])
                        push!(d_internal, vect[i][iii])
                    end
                    append!(d,d_internal)
                end
                vect = []
                push!(vect,d)
            end
        
            function number_to_tuples(vect)
                t = []
                for i in 1:3:length(vect)
                    push!(t,(vect[i],vect[i+1],vect[i+2]))
                end
                return t
            end
        
            return number_to_tuples(vect[1])
        end
        
        arr_heading = collect(1:map_params.nheading)
        arr_acceleration = collect(1:map_params.nacceleration)
        arr = [arr_heading, arr_acceleration]

        # Collect all combinations
        temp_arr = collect(Base.product(arr[1],arr[2]))    
        i = 3
        while_procedure_done = false
        while i <= length(arr)
            temp_arr = collect(Base.product(temp_arr,arr[i]))
            i += 1
            while_procedure_done = true
        end

        final_arr = vec(temp_arr)

        if while_procedure_done
            all_actions = remove_internal_tuples(final_arr)
        else
            all_actions = final_arr
        end
        
        return all_actions
    end

    global all_actions = as_map()
end

#-------------State Space Map (Q)
begin 
    #--------------------------Parameters: Object Contruction
    #Comment: The State Space Map is only needed for the Temporal Difference Agent
    struct ss_map_params
        nstates::Int64
        nvelocities::Int64
    end

    #--------------------------Define Object values
    function ss_map_params(;
        shipping_env_params = ShippingEnvParams(),
        nstates = shipping_env_params.gridworld_dims[1]*shipping_env_params.gridworld_dims[2],
        nvelocities = length(shipping_env_params.velocities),
        )
        ss_map_params(
            nstates,
            nvelocities,
        )
    end

    #--------------------------Final Mapping: Vector of all states
    function ss_map(;map_params = ss_map_params())

        function remove_internal_tuples(vect)
            d = []
            d_internal = []
            while first(vect)[1] isa Tuple
                d = []
                for i in 1:length(vect)
                    empty!(d_internal)
                    for ii in 1:length(first(vect[i]))
                        push!(d_internal,first(vect[i])[ii])
                    end
                    for iii in 2:length(vect[i])
                        push!(d_internal, vect[i][iii])
                    end
                    append!(d,d_internal)
                end
                vect = []
                push!(vect,d)
            end
        
            function number_to_tuples(vect)
                t = []
                for i in 1:3:length(vect)
                    push!(t,(vect[i],vect[i+1],vect[i+2]))
                end
                return t
            end
        
            return number_to_tuples(vect[1])
        end

        arr_states = collect(1:map_params.nstates)
        arr_velocities = collect(1:map_params.nvelocities)
        arr = [arr_states, arr_velocities]

        # Collect all combinations
        temp_arr = collect(Base.product(arr[1],arr[2]))
        while_procedure_done = false
        i = 3
        while i <= length(arr)
            temp_arr = collect(Base.product(temp_arr,arr[i]))
            i += 1
            while_procedure_done = true
        end

        #Create the final vector of all combinations

        final_arr = vec(temp_arr)

        if while_procedure_done
            all_actions = remove_internal_tuples(final_arr)
        else
            all_actions = final_arr
        end

        return all_actions
    end

    global all_states = ss_map()
end

#-------------Environment Contruction
begin
    mutable struct ShippingEnv <: AbstractEnv
        #Parameters
        params::ShippingEnvParams

        #Actions
        action_space::Base.OneTo{Int64}
        action::Int64 #action: (heading_angle,acceleration)

        #States      
        observation_space::Base.OneTo{Int64}
        state::Int64
        
        #Variables
        #---Captain Related
        position::CartesianIndex{2}
        velocity::Float32
        tAFT::Float32
        tFWD::Float32
        measurementTime::Float32 #counting up to 10 minutes before assigning measurement
        measurementSpeed:: Float32 #sum(speed_i*time_i)/measurementTime
        measurementDistance::Float32 #sum of step distances
        measurements::Int64
        lstmInputs::Matrix{Float32}

        #---Environment Related
        time::Float32
        distance::Float32 #Nautical Miles
        reward::Union{Nothing,Float32} 
        done::Bool 
        step::Int64
    end
    
    function ShippingEnv()
        env_parameters = ShippingEnvParams()
        env = ShippingEnv(
            #Parameters
            env_parameters,
    
            #Actions
            #action space
            Base.OneTo(length(all_actions)),
            #action
            rand(1:length(all_actions)),

            #States
            #observation space
            Base.OneTo(length(all_states)),
    
            #state
            LinearIndices((env_parameters.gridworld_dims[1],env_parameters.gridworld_dims[2]))[env_parameters.StartingPoint],
    
            #Variables
            #---Captain Related
            env_parameters.StartingPoint, #Position
            env_parameters.velocities[1], #Velocity
            15.2, #Draught AFT
            15.1, #Draught FWD
            0.0,
            0.0,
            0.0,
            0,
            rand(Float32,3,5),

            #---Environment Related
            0.0,
            0.0,
            0.0,
            false,  
            0
        )
        reset!(env)
        env
    end
end

#-------------Minimal interfaces implemented
begin
    RLBase.action_space(m::ShippingEnv) = m.action_space
    RLBase.state_space(m::ShippingEnv) = m.observation_space
    RLBase.reward(m::ShippingEnv) = m.done ? 0.0 : m.reward 
    RLBase.is_terminated(m::ShippingEnv) = m.done 
    RLBase.state(m::ShippingEnv) = m.state
    Random.seed!(31)

    function RLBase.reset!(m::ShippingEnv)
        m.position = m.params.StartingPoint
        m.velocity = m.params.velocities[1]
        m.done = false
        m.time = 0
        m.distance = 0
        m.reward = 0
        m.step = 0
        m.measurementDistance = 0.0
        m.measurementSpeed = 0.0
        m.measurementTime = 0.0
        m.measurements = 0
    end
end

#-------------Next step
begin
    function (m::ShippingEnv)(a::Int64)
        nextstep(m,all_actions[a][1],all_actions[a][2])
    end

    function nextstep(m::ShippingEnv, head_action, acceleration)
        
        # 1. Count Total Steps
        m.step += 1

        # 2. Course
        #   a. Note down the previous and the current state.
        heading = m.params.heading[head_action]
        thisState_norm = (m.position[1]/m.params.gridworld_dims[1],m.position[2]/m.params.gridworld_dims[2])
        m.position += heading
        nextState_norm = (m.position[1]/m.params.gridworld_dims[1],m.position[2]/m.params.gridworld_dims[2]) #normalized for going inanypolygon
        
        #   b. Calculate the distance covered in nautical miles.
        stepDistance = stepDistanceF(thisState_norm,nextState_norm)
        m.distance += stepDistance
        m.measurementDistance += stepDistance

        accident = false
        #   c. Check if ship crashes into boundaries.
        if m.position[1]<1 || m.position[1]>m.params.gridworld_dims[1] || m.position[2]<1 || m.position[2]>m.params.gridworld_dims[2] || inanypolygon(nextState_norm, m.params.all_polygons)
            m.position -= heading
            m.distance -= stepDistance
            accident = true
        end

        # 3. Speed & Acceleration
        #current_acceleration = m.params.acceleration[acceleration] #actual accelaration
        current_acceleration = m.params.acceleration[acceleration]
        if (m.velocity + current_acceleration) > minimum(m.params.velocities) && (m.velocity + current_acceleration < maximum(m.params.velocities))
            m.velocity += current_acceleration #-2 is used because accelaration input is 1-3 and we want to either go to lower acceleration or greater
        end

        # 4. Weather Construction
        weatherData = weatherConstruction(thisState_norm)
        significantWaves_height = weatherData

        # 5. State Definition
        m.state = LinearIndices((m.params.gridworld_dims[1],m.params.gridworld_dims[2]))[m.position] + m.params.gridworld_dims[1]*m.params.gridworld_dims[2]*first(findall(x->x==m.velocity,m.params.velocities))
        
        # 6. Time
        stepTime = stepDistance/m.velocity
        m.time += stepTime #hours
        m.measurementTime += stepTime

        m.measurementSpeed += m.velocity*stepTime

        # 7. Reward Calculation
        #   a. Initialize inputs vector

        m.reward = -0.001

        if m.measurementTime > (1/6) 
            inputVector = convert(Matrix{Float32},[m.measurementSpeed/m.measurementTime m.tAFT m.tFWD m.measurementDistance significantWaves_height])
            inputVector = (inputVector - meanInputs)./stdInputs
            m.measurements += 1

            m.lstmInputs = vcat(m.lstmInputs,inputVector)

            if m.measurements > 2
                Flux.reset!(model)
                result = [model(m.lstmInputs[x,:]) for x in (size(m.lstmInputs)[1]-2):size(m.lstmInputs)[1]]
                m.reward = -(stepTime/24)*abs((first(last(result))*stdOutput+meanOutput))
            end
    
            m.measurementDistance = 0.0
            m.measurementSpeed = 0.0
            m.measurementTime = 0.0
        end
        
        if accident 
            m.reward -= (1/450)*abs(stepDistanceF(thisState_norm,(m.params.GoalPoint[1]/m.params.gridworld_dims[1],m.params.GoalPoint[2],m.params.gridworld_dims[2])))
        end

        # 8. Episode Termination
        #   a. If goal is achieved
        m.done = m.position == m.params.GoalPoint

        #   b. If time > ETA
        if m.time > m.params.ETA
            m.reward -= (10/450)*abs(stepDistanceF(thisState_norm,(m.params.GoalPoint[1]/m.params.gridworld_dims[1],m.params.GoalPoint[2],m.params.gridworld_dims[2])))
            m.done = true
        end

    end

    function stepDistanceF(normPosPrev,normPosNext;
        maxLat = -6.6123212, 
        minLat = -7.0099144, 
        maxLon = 62.0226423, 
        minLon = 61.3895367)
    
        lmbda = (maxLat-minLat)/(maxLon-minLon)
    
        positionPreviousLat = (normPosPrev[1]-0.25)*(maxLat-minLat)/lmbda + minLat
        positionPreviousLon = normPosPrev[2]*(maxLon-minLon) + minLon
    
        positionNextLat = (normPosNext[1]-0.25)*(maxLat-minLat)/lmbda + minLat
        positionNextLon = normPosNext[2]*(maxLon-minLon) + minLon
    
        phi1 = deg2rad(positionPreviousLat)
        phi2 = deg2rad(positionNextLat)
        Dphi = phi2-phi1
        Dlambda = deg2rad(positionNextLon-positionPreviousLon)
    
        a = (sin(Dphi/2))^2 + cos(phi1)*cos(phi2)*(sin(Dlambda/2)^2)
        c = 2*atan(sqrt(a),sqrt(1-a))
    
        # Distance calculated in nautical miles
        d = 6731*c*0.539957
    
        return d
    
    end
end

#------------- Weather Construction Function
begin 
    global centerStorm = [0.7,0.5]
    global radiusStorm = 0.2
    global badWeather_SWHR = 3.5
    global calmWeather_SWHR = 0.5

    function weatherConstruction(position;
        stormCenter = centerStorm,
        stormRadius = radiusStorm, 
        BWstormWaves_height=badWeather_SWHR,
        CWstormWaves_height=calmWeather_SWHR)
        distanceSquared = (stormCenter[1]-position[1])^2 + (stormCenter[2]-position[2])^2
        if distanceSquared < stormRadius^2
            significantWaves_height = BWstormWaves_height
        else 
            significantWaves_height = CWstormWaves_height
        end
        return significantWaves_height

    end
end

#------------- Environment
env = ShippingEnv()
RLBase.test_runnable!(env)

#------------- Hook
begin
    Base.@kwdef mutable struct customizedHook <: AbstractHook
        velocity::Vector{Int8} = []
        velocity_total::Vector{Vector{Int8}} = []
        position:: Vector{CartesianIndex{2}} = []
        position_total:: Vector{Vector{CartesianIndex{2}}} = []
        reward::Vector{Float32} = []
        reward_total::Vector{Vector{Float32}} =[]
    end

    (h::customizedHook)(::PostActStage,agent,env) = 
    begin 
        push!(h.velocity,env.velocity)
        push!(h.position,env.position)
        push!(h.reward,env.reward)
    end

    (h::customizedHook)(::PostEpisodeStage,agent,env) = 
    begin 
        h.velocity_total = vcat(h.velocity_total,[h.velocity])
        h.position_total = vcat(h.position_total,[h.position])
        h.reward_total = vcat(h.reward_total,[h.reward])
    end

    (h::customizedHook)(::PreEpisodeStage,agent,env) = 
    begin 
        h.velocity = []
        h.position = []
        h.reward = []
    end
end

hook = customizedHook();

agentQ = Agent(
    policy=QBasedPolicy(
        learner=TDLearner(
            approximator=TabularQApproximator(
                ;n_state=length(state_space(env)),
                n_action=length(action_space(env)),
                opt=Descent(0.3)
                ),
            method=:SARSA
        ),
        explorer=EpsilonGreedyExplorer(kind = :linear, 
        step = 1, #record the current step
        ϵ_init = 0.99, #initial epsilon
        warmup_steps = 10000, #the number of steps to use ϵ_init
        decay_steps = 5000, #the number of steps for epsilon to decay from ϵ_init to ϵ_stable
        ϵ_stable = 0.1, #the epislon after warmup_steps + decay_steps
        is_break_tie = true)
    ),
    #A trajectory is the sequence of what has happened over a set of continuous timestamps
    #trajectory = Vector of Vectors [1. state 2. action 3. reward 4. terminal]
    trajectory=VectorSARTTrajectory()
)

#------------- Run Experiment
stopCondition =  StopAfterEpisode(1500, is_show_progress = true);
ex = Experiment(agentQ, env, stopCondition, hook, "#Test")

@time run(ex)

#------------- Results Evaluation
mean_reward(ex.hook.reward_total)
plotRewards(ex.hook.reward_total)
plotMeanVelocities(ex.hook.velocity_total)
plotRoute(ex.hook.position_total[end],env)

p = [sum(ex.hook.reward_total[i]) for i in 1:length(ex.hook.reward_total)]
maximum(p)
k = argmax(p)
plotRoute(ex.hook.position_total[k],env)
plotVelocityProfile(ex.hook.velocity_total[k],env.time)

#------------- Results Evaluation Functions
begin 
    titleFont = (10,"Computer Modern");
    selectedFont = "Computer Modern";
    DPI = 500;
    guideFontSize = 8;
end

function plotRoute(positions,env)

    function circleShape(h,k,r)
        theta = LinRange(0,2*pi,500)
        h.+r*sin.(theta), k.+r*cos.(theta)
    end

    # Store Position Values
    pos = Vector{CartesianIndex}[]
    x = Vector{Int}[]
    y = Vector{Int}[]
    pos = vcat(pos,env.params.StartingPoint)
    x = vcat(x,pos[1][1])
    y = vcat(y,pos[1][2])
    for i in 1:length(positions)
        pos = vcat(pos,positions[i])
        x = vcat(x,pos[i+1][1])
        y = vcat(y,pos[i+1][2])
    end

    # Store Direction Values
    x_dir = Vector{Int}[]
    y_dir = Vector{Int}[]
    for k in 2:(length(pos))
        x_dir = vcat(x_dir,x[k]-x[k-1])
        y_dir = vcat(y_dir,y[k]-y[k-1])
    end

    polygons = GeoBoundariesManipulation.load_files("normalized_polygons")
    p = GeoBoundariesManipulation.overall_plot(env.params.all_polygons,polygons,env.params.gridworld_dims[1],env.params.gridworld_dims[2])
    quiver!(p,x[1:end-1]/env.params.gridworld_dims[1],y[1:end-1]/env.params.gridworld_dims[2],quiver=(x_dir/env.params.gridworld_dims[1],y_dir/env.params.gridworld_dims[2]))
    scatter!(p,(env.params.StartingPoint[1]/env.params.gridworld_dims[1],env.params.StartingPoint[2]/env.params.gridworld_dims[2]), label = "StartingPoint")
    scatter!(p,(env.params.GoalPoint[1]/env.params.gridworld_dims[1],env.params.GoalPoint[2]/env.params.gridworld_dims[2]), label = "GoalPoint")
    plot!(p,circleShape(centerStorm[1],centerStorm[2],radiusStorm),seriestype=[:shape],color="red",fillcolor="red",linecolor="red",fillalpha=0.3,label="Storm")

    return plot(p,
    title="Ship's Course",
    titlefont = titleFont,
    tickfontfamily = selectedFont,
    dpi = DPI,
    axis=false)
end

function mean_reward(total_reward)
    totalreward = []
    for i in 1:length(total_reward)
        push!(totalreward,sum(total_reward[i]))
    end
    return sum(totalreward)/length(totalreward) 
end

function plotRewards(total_reward)
    rewarding = []
    for i in 1:length(total_reward) 
        push!(rewarding,sum(total_reward[i]))
    end
    return plot(rewarding,
    label=nothing,
    xlabel="Episodes",
    title="Rewards per Episode \n If reaches Goal Point equals to Tons of Fuel Consumed Per Episode",
    titlefont=titleFont,
    guidefont=selectedFont,
    tickfontfamily = selectedFont,
    guidefontsize = guideFontSize,
    dpi = DPI)
end

function plotMeanVelocities(total_velocities)
    velocities = []
    for i in 1:length(total_velocities) 
        push!(velocities,sum(total_velocities[i])/length(total_velocities[i]))
    end
    return plot(velocities,
    label=nothing,
    xlabel = "Episodes",
    title="Mean Velocity per Episode",
    titlefont = titleFont,
    guidefont = selectedFont,
    tickfontfamily = selectedFont,
    guidefontsize = guideFontSize,
    dpi = DPI)
end

function plotVelocityProfile(velocity,time)
    plot(velocity,title=string("Velocity profile \n","Time:     ",round(time;digits=3)," hours"),titlefont=titleFont,guidefont=selectedFont,dpi=DPI,label=nothing)
end