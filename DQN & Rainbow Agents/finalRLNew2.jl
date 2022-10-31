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
    using CUDA
    using Optim
end

#The following custom package is only used in order to 
#load the boundary data for the environment and check if 
#the next position the agent tries to move to is inside any of the boundaries
include(joinpath(pwd(),"GeoBoundariesManipulation.jl"));
using .GeoBoundariesManipulation

#-------------Load LSTM Model
@load "lstmModel.bson" m
model = deepcopy(m) |> gpu

begin 
    global meanInputs = permutedims(convert(Vector{Float32},readdlm("inputsMeanDeviation.txt")[:,1]))
    global stdInputs = permutedims(convert(Vector{Float32},readdlm("inputsMeanDeviation.txt")[:,2]))

    global meanOutput = first(readdlm("labelMeanDeviation.txt")[:,1])
    global stdOutput = first(readdlm("labelMeanDeviation.txt")[:,2])

    randomSeed = MersenneTwister(22)
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
        gridworld_dims = (15,15),
        velocities = Vector((6.0:2.0:12.0)), #knots
        acceleration = Vector((-2.0:2.0:2.0)), 
        heading = [CartesianIndex(0,1);CartesianIndex(0,-1);CartesianIndex(-1,0);CartesianIndex(-1,1);CartesianIndex(-1,-1);CartesianIndex(1,-1);CartesianIndex(1,1);CartesianIndex(1,0)], 
        StartingPoint = GeoBoundariesManipulation.GoalPointToCartesianIndex((-6.733535,61.997345),gridworld_dims[1],gridworld_dims[2]),
        EndingPoint = GeoBoundariesManipulation.GoalPointToCartesianIndex((-6.691500,61.535580),gridworld_dims[1],gridworld_dims[2]),
        AllPolygons = GeoBoundariesManipulation.load_files("finalboundaries"),
        eta = 3.5 #hours
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

#-------------Environment Contruction
begin
    mutable struct ShippingEnv <: AbstractEnv
        #Parameters
        params::ShippingEnvParams

        #Actions
        action_space::Base.OneTo{Int64}
        action::Int64 #action: (heading_angle,acceleration)

        #States
        
        observation_space::Space{Vector{Interval{Int64, Closed, Closed}}}
        state::Vector{Float64}
        
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
        lstmInputs::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}

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
            Space([0..1,0..1,0..1]),
    
            #state
            [env_parameters.StartingPoint[1],env_parameters.StartingPoint[2],env_parameters.velocities[1]],
    
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
            CuArray(rand(Float32,3,5)),

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

#-------------State Normalization
function DQN_stateNormalization(m::ShippingEnv)
    max_st_position = m.params.gridworld_dims[1]
    min_st_position = 1

    max_st_velocity = maximum(m.params.velocities)
    min_st_velocity = minimum(m.params.velocities)

    position_x = (m.state[1] - min_st_position)/(max_st_position-min_st_position)
    position_y = (m.state[2] - min_st_position)/(max_st_position-min_st_position)

    velocity = (m.state[3] - min_st_velocity)/(max_st_velocity-min_st_velocity)

    return CuArray([position_x,position_y,velocity])

end

#-------------Minimal interfaces implemented
begin
    RLBase.action_space(m::ShippingEnv) = m.action_space
    RLBase.state_space(m::ShippingEnv) = m.observation_space
    RLBase.reward(m::ShippingEnv) = m.done ? 0.0 : m.reward
    RLBase.is_terminated(m::ShippingEnv) = m.done 
    RLBase.state(m::ShippingEnv) = DQN_stateNormalization(m)
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
        m.lstmInputs = CuArray(rand(Float32,3,5))
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
        current_acceleration = m.params.acceleration[acceleration] #actual accelaration
        if (m.velocity + current_acceleration) > minimum(m.params.velocities) && (m.velocity + current_acceleration < maximum(m.params.velocities))
            m.velocity += current_acceleration #-2 is used because accelaration input is 1-3 and we want to either go to lower acceleration or greater
        end

        # 4. Weather Construction
        weatherData = weatherConstruction(thisState_norm)
        significantWaves_height = weatherData 

        # 5. State Definition
        positionX = Int(m.position[1])
        positionY = Int(m.position[2])
        m.state = [positionX,positionY,m.velocity]

        # 6. Time
        stepTime = stepDistance/m.velocity
        m.time += stepTime
        m.measurementTime += stepTime

        m.measurementSpeed += m.velocity*stepTime

        # 7. Reward Calculation
        #   a. Initialize inputs vector

        m.reward = -0.001

        if m.measurementTime > (1/6) 
            inputVector = convert(Matrix{Float32},[m.measurementSpeed/m.measurementTime m.tAFT m.tFWD m.measurementDistance significantWaves_height])
            inputVector = CuArray((inputVector - meanInputs)./stdInputs)
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
        if m.step > 1000
            m.reward -= (1/450)*abs(stepDistanceF(thisState_norm,(m.params.GoalPoint[1]/m.params.gridworld_dims[1],m.params.GoalPoint[2],m.params.gridworld_dims[2])))
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
    global calmWeather_SWHR = 0.7

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

#------------- DQN Agent Construction
begin 
    function nnModel(hidden1,hidden2,hidden3,input,output,activation)
        return Chain(
            Dense(input,hidden1,activation),
            Dense(hidden1,hidden2,activation),
            Dense(hidden2,hidden3,activation),
            Dense(hidden3,output)
        )
    end

    function DQN_agentConstruction(;
        hidden1=40, 
        hidden2=50, 
        hidden3=20,
        updhor = 2,
        update_frequen = 4,
        batchsize = 100,
        minimumreplayhistory = 100,
        targetupdate = 100,
        warmupsteps=0,
        decaysteps = 0,
        trajectorycapacity = 200,
        learningrate= 0.001,
        discountrate = 0.99f0,
        gmomentum = 0.95,
        epsilon_final = 0.1,
        activationfunction = relu)

        agent = Agent(
            policy=QBasedPolicy(
                #DQNLearner will be used because of the option for double dqn.
                learner=DQNLearner( 
                    approximator=NeuralNetworkApproximator(
                        model = nnModel(hidden1,hidden2,hidden3,length(state(env)),length(action_space(env)),activationfunction) |> gpu,
                        optimizer = ADAM(learningrate, (gmomentum,0.999)), 
                    ),
                    target_approximator = NeuralNetworkApproximator(
                        model = nnModel(hidden1,hidden2,hidden3,length(state(env)),length(action_space(env)),activationfunction) |> gpu,
                        optimizer = ADAM(learningrate, (gmomentum,0.999)), 
                    ),
                    loss_func = Flux.huber_loss,
                    γ = discountrate, #discount rate
                    batch_size = batchsize, #mini batch_size
                    update_horizon = updhor, #G = r .+ γ^n .* (1 .- t) .* q′
                    #---min_replay_history
                    #number of transitions that should be made before updating the approximator
                    #it is the replay_start_size = the count of experiences (frames) to add to replay buffer before starting training
                    min_replay_history = minimumreplayhistory, 
                    update_freq = update_frequen, #the frequency of updating the approximator
                    #---target_update_freq 
                    #how frequently we sync model weights from the main DQN network to the target DQN network
                    #(how many frames in between syncing) 
                    target_update_freq = targetupdate, 
                    stack_size = nothing, #use the recent stack_size frames to form a stacked state
                    traces = SARTS, #current state, action, reward, terminal, next state
                    #rng = randomSeed,
                    is_enable_double_DQN = true #enable double dqn, enabled by default
                ),
                explorer = EpsilonGreedyExplorer(;
                    kind = :linear, 
                    step = 1, #record the current step
                    ϵ_init = 0.99, #initial epsilon
                    warmup_steps = warmupsteps, #the number of steps to use ϵ_init
                    decay_steps = decaysteps, #the number of steps for epsilon to decay from ϵ_init to ϵ_stable
                    ϵ_stable = epsilon_final, #the epislon after warmup_steps + decay_steps
                    is_break_tie = true, #randomly select an action of the same maximum values if set to true.
                    #rng = randomSeed, #set the internal rng
                    is_training = true #in training mode, step will not be updated and the epsilon will be set to 0. 
                )
            ),
            #A trajectory is the sequence of what has happened over a set of continuous timestamps
            trajectory=CircularArraySARTTrajectory(;
                capacity = trajectorycapacity,
                state = Vector{Float64} => (length(state(env)),),
                # action = Int => (),
                # reward = Float32 => (),
                # terminal = Bool => (),
            ) #when using NN you have to change from VectorSARTTrajectory to CircularArraysTraject
        )

        return agent
    end

    function rainbowDQN_agentConstruction(;
        hidden1=20, 
        hidden2=20, 
        hidden3=20,
        natoms = 51,
        update_frequen = 4,
        batchsize = 100,
        minimumreplayhistory = 100,
        targetupdate = 100,
        warmupsteps = 0,
        decaysteps = 0,
        trajectorycapacity = 200,
        learningrate= 0.001,
        discountrate = 0.99f0,
        gmomentum = 0.95,
        epsilon_final = 0.1,
        activationfunction = relu)

        agent = Agent(
            policy=QBasedPolicy(
                #DQNLearner will be used because of the option for double dqn.
                learner=RainbowLearner( 
                    approximator=NeuralNetworkApproximator(
                        model = nnModel(hidden1,hidden2,hidden3,length(state(env)),length(action_space(env))*natoms,activationfunction) |> gpu,
                        optimizer = ADAM(learningrate, (gmomentum,0.999)), 
                    ),
                    target_approximator = NeuralNetworkApproximator(
                        model = nnModel(hidden1,hidden2,hidden3,length(state(env)),length(action_space(env))*natoms,activationfunction) |> gpu,
                        optimizer = ADAM(learningrate, (gmomentum,0.999)), 
                    ),
                    n_actions = length(action_space(env)),
                    n_atoms = natoms,
                    Vₘₐₓ = +100.0f0,
                    Vₘᵢₙ = -100.0f0,
                    update_freq = update_frequen,
                    γ = discountrate,
                    batch_size = batchsize,
                    stack_size = nothing,
                    min_replay_history = minimumreplayhistory,
                    loss_func = Flux.logitcrossentropy,
                    target_update_freq = targetupdate

                ),
                explorer = EpsilonGreedyExplorer(;
                    kind = :linear, 
                    step = 1, #record the current step
                    ϵ_init = 0.99, #initial epsilon
                    warmup_steps = warmupsteps, #the number of steps to use ϵ_init
                    decay_steps = decaysteps, #the number of steps for epsilon to decay from ϵ_init to ϵ_stable
                    ϵ_stable = epsilon_final, #the epislon after warmup_steps + decay_steps
                    is_break_tie = true, #randomly select an action of the same maximum values if set to true.
                    #rng = randomSeed, #set the internal rng
                    is_training = true #in training mode, step will not be updated and the epsilon will be set to 0. 
                )
            ),
            #A trajectory is the sequence of what has happened over a set of continuous timestamps
            trajectory=CircularArraySARTTrajectory(;
                capacity = trajectorycapacity,
                state = Vector{Float64} => (length(state(env)),),
                # action = Int => (),
                # reward = Float32 => (),
                # terminal = Bool => (),
            ) #when using NN you have to change from VectorSARTTrajectory to CircularArraysTraject
        )

        return agent
    end
end

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

agentDQN = DQN_agentConstruction(;
    #nn
    hidden1=35,
    hidden2=50,
    hidden3=35,
    learningrate= 0.75*10^(-5),
    gmomentum = 0.9,
    activationfunction = relu,

    #dqn
    updhor=4,
    discountrate = 0.99f0,
    update_frequen = 16,
    batchsize = 2,
    minimumreplayhistory = 500,
    targetupdate = 170,
    trajectorycapacity = 1500,
    
    #explorer
    warmupsteps= 50000,
    decaysteps = 10000,
    epsilon_final = 0.1
)

agentRainbow = rainbowDQN_agentConstruction(;
    #nn
    hidden1=40, 
    hidden2=35, 
    hidden3=25,
    activationfunction = relu,
    learningrate= 50*10^(-5),
    gmomentum = 0.9,

    #rainbow dqn
    natoms = 51,
    update_frequen = 6,
    targetupdate = 325,
    discountrate = 0.99f0,
    batchsize = 150,
    minimumreplayhistory = 500,
    trajectorycapacity = 1280,


    #explorer
    warmupsteps = 30000,
    decaysteps = 10000,
    epsilon_final = 0.1
)

#------------- Run Experiment
stopCondition =  StopAfterEpisode(800, is_show_progress = true);
ex = Experiment(agentRainbow, env, stopCondition, hook, "#Test")

@time run(ex)

#------------- Results Evaluation
mean_reward(ex.hook.reward_total[1:end])
plotRewards(ex.hook.reward_total[1:end])
plotMeanVelocities(ex.hook.velocity_total)
plotRoute(ex.hook.position_total[end],env)

p = [sum(ex.hook.reward_total[i]) for i in 1:length(ex.hook.reward_total)]
maximum(p)
k = argmax(p)
plotRoute(ex.hook.position_total[k],env)

#------------- Hyperopt
ho = @hyperopt for i = 50,
        sampler = RandomSampler(),
        h1 = collect(10:5:60),
        h2 = collect(10:5:60),
        h3 = collect(10:5:60),
        lr = exp10.(LinRange(-6,-3,300)),
        updateFreq = collect(2:2:50),
        updateTarget = collect(100:5:500),
        batchSize = collect(2:4:150),
        trajectoryCapacity = collect(500:20:2000),
        updHor = collect(2:2:10)

    print("\n","Sample Number:","\t", i,"\n")
    print("Hidden Layers: ",h1,",",h2,",",h3,"\n")
    print("Learning Rate: ",lr,"\n")
    print("Update: ",updateFreq,"\t","Target Update: ",updateTarget,"\t","Update Horizon: ",updHor,"\n")
    print("Batch Size: ",batchSize,"\t","Trajectory Capacity: ",trajectoryCapacity,"\n")
    @show f(h1,h2,h3,0.65*10^(-5),updateFreq,updateTarget,batchSize,trajectoryCapacity,updHor;i)
end

ho = @hyperopt for i = 100,
    sampler = RandomSampler(),
    h1 = collect(10:5:60),
    h2 = collect(10:5:60),
    h3 = collect(10:5:60),
    lr = exp10.(LinRange(-6,-3,300)),
    updateFreq = collect(2:2:50),
    updateTarget = collect(100:5:500),
    batchSize = collect(2:4:150),
    trajectoryCapacity = collect(500:20:2000),
    updHor = collect(2:2:10)

print("\n","Sample Number:","\t", i,"\n")
print("Hidden Layers: ",h1,",",h2,",",h3,"\n")
print("Learning Rate: ",lr,"\n")
print("Update: ",updateFreq,"\t","Target Update: ",updateTarget,"\t","Update Horizon: ",updHor,"\n")
print("Batch Size: ",batchSize,"\t","Trajectory Capacity: ",trajectoryCapacity,"\n")
@show f(h1,h2,h3,lr,updateFreq,updateTarget,batchSize,trajectoryCapacity,updHor;i)
end

function f(h1,h2,h3,lr,updFreq,updTarg,batchSize,trajCap,updHor;i)

    env = ShippingEnv();

    #agentDQN = DQN_agentConstruction(;
    #nn
    #hidden1=h1,
    #hidden2=h2,
    #hidden3=h3,
    #learningrate= lr,
    #gmomentum = 0.9,
    #activationfunction = relu,

    #dqn
    #updhor=updHor,
    #discountrate = 0.99f0,
    #update_frequen = updFreq,
    #batchsize = batchSize,
    #minimumreplayhistory = 500,
    #targetupdate = updTarg,
    #trajectorycapacity = trajCap,
    
    #explorer
    #warmupsteps= 50000,
    #decaysteps = 10000,
    #epsilon_final = 0.1
    #)

    agentRainbow = rainbowDQN_agentConstruction(;
    #nn
    hidden1=h1, 
    hidden2=h2, 
    hidden3=h3,
    activationfunction = relu,
    learningrate= lr,
    gmomentum = 0.9,

    #rainbow dqn
    natoms = 51,
    update_frequen = updFreq,
    targetupdate = updTarg,
    discountrate = 0.99f0,
    batchsize = batchSize,
    minimumreplayhistory = 500,
    trajectorycapacity = trajCap,


    #explorer
    warmupsteps = 30000,
    decaysteps = 10000,
    epsilon_final = 0.1
    )   


    hook = customizedHook();

    stop_condition =  StopAfterEpisode(700, is_show_progress = true)

    ex = Experiment(agentRainbow, env, stop_condition, hook, "#Test")

    @time run(ex)

    p = plot(plotRewards(ex.hook.reward_total),title=string("h1=",h1,"  | h2=",h2,"   | h3=",h3,"\n","lr (x10^5)=",round(lr*10^5;digits=4),"     updFreq=",updFreq,"    updTarg=",updTarg,"\n","bS=",batchSize,"     Capacity=",trajCap,"       UpdHor=",updHor))

    savefig(p,string("C:\\Users\\LME-lab\\Desktop\\hyperoptFinal\\",i))

    display(p)

    return mean(sum(ex.hook.reward_total[ii]) for ii in 10:length(ex.hook.reward_total))
end

printmax(ho)
plot(ho)

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

    polygons = GeoBoundariesManipulation.load_files("boundaries")
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
        title="Rewards per episode \n If agent reaches the goal point, the reward is equal to TFOC per episode",
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

#-------------Saving & Loading NN Parameters
begin
    weights_app = Flux.params(ex.policy.policy.learner.approximator.model);
    dst_app = ex.policy.policy.learner.approximator.model;

    weights_tar = Flux.params(ex.policy.policy.learner.target_approximator.model);
    dst_tar = ex.policy.policy.learner.target_approximator.model;

    BSON.@save "wandb.bson" weights_app
    BSON.@load "wandb.bson" weights_app

    BSON.@save "wandbtar.bson" weights_tar
    BSON.@load "wandbtar.bson" weights_tar

    Flux.loadparams!(dst_app,weights_app)
    Flux.loadparams!(dst_tar,weights_tar)
end

# --- Function to calculate stepDistance
# --- The results of the following have been used. 

polygData = readdlm(joinpath(pwd(),"boundaries",filelist[1]))
file_list = readdir(joinpath(pwd(),"boundaries"))

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

optimas = optimums(file_list)

maxLat = optimas[1]
minLat = optimas[2]
maxLon = optimas[3]
minLon = optimas[4]

stepDistanceF((0.52,0.52),(0.54,0.54))