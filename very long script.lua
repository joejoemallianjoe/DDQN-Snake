local NeuralNetworkData = {}
local DerivativeFunctionLib = {
	['SigmoidDerivative'] = function(y)
		local function Sigmoid(x)
			return 1 / (1 + math.exp(-x))
		end

		return Sigmoid(y) * (1 - Sigmoid(y))
	end,
	['ReLUDerivative'] = function(x)
		if x > 0 then
			return 1
		else
			return 0
		end
	end,
	['LeakyReLUDerivative'] = function(x)
		if x > 0 then
			return 1
		else
			return 0.00001
		end
	end,
	['MeanSquaredErrorDerivative'] = function(OutputVector,TargetVector)
		local ReturnVector = {}

		for i,v in ipairs(TargetVector) do
			local o = OutputVector[i]
			table.insert(ReturnVector,o - v)
		end

		return ReturnVector
	end,
	['TanHDerivative'] = function(x) -- NOTE TO FUTURE SELF: WATCH OUT FOR WEIRD EDGE CASES LIKE INF / INF OR -INF / INF!
		return 1 - math.tanh(x)^2
	end,
	['NoneDerivative'] = function(x)
		return 1
	end,
}

local FunctionLib = {
	['Sigmoid'] = function(x)
		return 1 / (1 + math.exp(-x))
	end,
	['ReLU'] = function(x)
		if x > 0 then
			return x
		else
			return 0
		end
	end,
	['LeakyReLU'] = function(x)
		if x > 0 then
			return x
		else
			return 0.00001 * x
		end
	end,
	['TanH'] = function(x)
		return math.tanh(x)
	end,
	['MeanSquaredError'] = function(OutputVector,TargetVector)
		local ReturnVector = {}

		for i,v in ipairs(TargetVector) do
			local o = OutputVector[i]
			table.insert(ReturnVector,0.5 * (v - o)^2)
		end

		return ReturnVector
	end,
	['MatrixMultiplication'] = function(MatrixA,MatrixB)
		local ResultMatrix = {}

		for i = 1,#MatrixA do
			local z = {}

			for j = 1,#MatrixB[1] do
				table.insert(z,0)
			end

			table.insert(ResultMatrix,z)
		end

		if #MatrixA[1] ~= #MatrixB then
			assert(true,'Dimensions are incompatible!')
		else
			for MatrixARowIndex = 1,#MatrixA do
				for MatrixBColumnIndex = 1,#MatrixB[1] do
					for MatrixBRowIndex = 1,#MatrixB do
						ResultMatrix[MatrixARowIndex][MatrixBColumnIndex] += MatrixA[MatrixARowIndex][MatrixBRowIndex] * MatrixB[MatrixBRowIndex][MatrixBColumnIndex]
					end
				end
			end

			return ResultMatrix
		end
	end
}

local NeuralNetworkFramework = {
	['CreateNN'] = function(Name,LearningRate,NumInputs,NumHiddenNodes,NumHiddenLayers,NumOutputs,WeightMin,WeightMax,HiddenLayerActivationFunction,OutputLayerActivationFunction,Optimizer,ExtraParams)
		if NeuralNetworkData[Name] ~= nil then return end

		local Rand = Random.new()
		local NumLayers = NumHiddenLayers + 2

		NeuralNetworkData[Name] = {
			['NumLayers'] = NumLayers,
			['Layer1'] = {
				['ActivationFunction'] = 'None',
				['LayerType'] = 'First',
				['Nodes'] = {}
			}
		}

		NeuralNetworkData[Name].Optimizer = Optimizer

		if string.match(Optimizer,'Minibatch') ~= nil then
			NeuralNetworkData[Name].BatchSize = ExtraParams.BatchSize
		end

		NeuralNetworkData[Name].LearningRate = LearningRate
		NeuralNetworkData[Name].OriginalLearningRate = LearningRate
		NeuralNetworkData[Name].NumGradientSteps = 0

		if string.match(Optimizer,'Momentum') ~= nil then
			NeuralNetworkData[Name].Momentum = ExtraParams.Momentum
		end

		for i = 1,NumInputs do
			NeuralNetworkData[Name].Layer1.Nodes['Node'..tostring(i)] = {
				['Value'] = 0,
				['Connections'] = {

				}
			}
		end

		for i = 1,NumInputs do
			if NumHiddenNodes ~= 0 then
				for Index = 1,NumHiddenNodes do
					table.insert(NeuralNetworkData[Name].Layer1.Nodes['Node'..tostring(i)].Connections,{Weight = Rand:NextNumber(WeightMin,WeightMax),ConnectedLayer = 'Layer2',ConnectedNode = 'Node'..tostring(Index),Gradient = 0,PrevVelocity = 0})
				end
			else
				for Index = 1,NumOutputs do
					table.insert(NeuralNetworkData[Name].Layer1.Nodes['Node'..tostring(i)].Connections,{Weight = Rand:NextNumber(WeightMin,WeightMax),ConnectedLayer = 'Layer2',ConnectedNode = 'Node'..tostring(Index),Gradient = 0,PrevVelocity = 0})
				end
			end
		end

		for i = 2,NumHiddenLayers + 1 do
			local LT = 'None'

			if HiddenLayerActivationFunction ~= 'None' then
				LT = 'ActivationLayer'
			end

			NeuralNetworkData[Name]['Layer'..tostring(i)] = {
				['ActivationFunction'] = HiddenLayerActivationFunction,
				['LayerType'] = LT,
				['Nodes'] = {}
			}

			local Ref = NeuralNetworkData[Name]['Layer'..tostring(i)].Nodes

			for j = 1,NumHiddenNodes do
				Ref['Node'..tostring(j)] = {
					['Value'] = 0,
					['Bias'] = Rand:NextNumber(WeightMin,WeightMax),
					['Connections'] = {

					},
					['WeightedSumVal'] = 0,
					['Gradient'] = 0,
					['BiasUpdated'] = false,
					['PrevVelocity'] = 0,
					['S'] = 0
				}

				if i == NumLayers - 1 then
					for k = 1,NumOutputs do
						table.insert(Ref['Node'..tostring(j)].Connections,{Weight = Rand:NextNumber(WeightMin,WeightMax),ConnectedLayer = 'Layer'..tostring(i + 1),ConnectedNode = 'Node'..tostring(k),Gradient = 0,PrevVelocity = 0})
					end
				else
					for k = 1,NumHiddenNodes do
						table.insert(Ref['Node'..tostring(j)].Connections,{Weight = Rand:NextNumber(WeightMin,WeightMax),ConnectedLayer = 'Layer'..tostring(i + 1),ConnectedNode = 'Node'..tostring(k),Gradient = 0,PrevVelocity = 0})
					end
				end
			end
		end

		NeuralNetworkData[Name]['Layer'..tostring(NumLayers)] = {
			['ActivationFunction'] = OutputLayerActivationFunction,
			['LayerType'] = 'Last',
			['Nodes'] = {}
		}

		local Ref = NeuralNetworkData[Name]['Layer'..tostring(NumLayers)].Nodes

		for i = 1,NumOutputs do
			Ref['Node'..tostring(i)] = {
				['Value'] = 0,
				['Bias'] = Rand:NextNumber(WeightMin,WeightMax),
				['WeightedSumVal'] = 0,
				['Gradient'] = 0,
				['BiasUpdated'] = false,
				['PrevVelocity'] = 0,
				['S'] = 0
			}
		end

		--warn(NeuralNetworkData[Name])
	end,
	['CalculateForwardPass'] = function(Name,InputVector)
		local Data = NeuralNetworkData[Name]

		local c = 0

		for i,v in pairs(Data.Layer1.Nodes) do
			c = c + 1
		end

		assert(#InputVector == c,'Wrong length of inputs! Did you forget to change NumInputs?')

		local OutputVals = {}

		for i = 2,Data.NumLayers do
			local Layer = 'Layer'..tostring(i)
			local LayerData = Data[Layer]

			for j,v in pairs(LayerData.Nodes) do
				v.Value = 0
			end
		end

		for i = 1,Data.NumLayers do
			local Layer = 'Layer'..tostring(i)
			local LayerData = Data[Layer]
			local CurrentNodeIndex = 0

			if LayerData.LayerType == 'First' then
				for NodeName,NodeData in pairs(LayerData.Nodes) do
					CurrentNodeIndex = CurrentNodeIndex + 1
					NodeData.Value = InputVector[CurrentNodeIndex]

					for i,ConnectionData in ipairs(NodeData.Connections) do
						local w = ConnectionData.Weight
						local ConnectedToLayer = ConnectionData.ConnectedLayer
						local ConnectedToNode = ConnectionData.ConnectedNode

						Data[ConnectedToLayer].Nodes[ConnectedToNode].Value = Data[ConnectedToLayer].Nodes[ConnectedToNode].Value + w * NodeData.Value
					end
				end
			elseif LayerData.LayerType == 'None' then
				for NodeName,NodeData in pairs(LayerData.Nodes) do
					CurrentNodeIndex = CurrentNodeIndex + 1
					NodeData.Value = NodeData.Value + NodeData.Bias
					NodeData.WeightedSumVal = NodeData.Value

					for i,ConnectionData in ipairs(NodeData.Connections) do
						local w = ConnectionData.Weight
						local ConnectedToLayer = ConnectionData.ConnectedLayer
						local ConnectedToNode = ConnectionData.ConnectedNode

						Data[ConnectedToLayer].Nodes[ConnectedToNode].Value = Data[ConnectedToLayer].Nodes[ConnectedToNode].Value + w * NodeData.Value
					end
				end
			elseif LayerData.LayerType == 'ActivationLayer' then
				--Add bias, then activate, then multiply by weight
				local NumNodes = 0

				for i,v in pairs(LayerData.Nodes) do
					NumNodes = NumNodes + 1
				end

				for i = 1,NumNodes do
					local NodeName = 'Node'..tostring(i)
					local NodeData = LayerData.Nodes[NodeName]
					CurrentNodeIndex = CurrentNodeIndex + 1

					NodeData.Value = NodeData.Value + NodeData.Bias
					NodeData.WeightedSumVal = NodeData.Value

					NodeData.Value = FunctionLib[LayerData.ActivationFunction](NodeData.Value)

					for i,ConnectionData in ipairs(NodeData.Connections) do
						local w = ConnectionData.Weight
						local ConnectedToLayer = ConnectionData.ConnectedLayer
						local ConnectedToNode = ConnectionData.ConnectedNode

						Data[ConnectedToLayer].Nodes[ConnectedToNode].Value = Data[ConnectedToLayer].Nodes[ConnectedToNode].Value + w * NodeData.Value
					end
				end
			else
				local NumNodes = 0

				for i,v in pairs(LayerData.Nodes) do
					NumNodes = NumNodes + 1
				end

				-- LayerType = final
				for i = 1,NumNodes do
					local NodeName = 'Node'..tostring(i)
					local NodeData = LayerData.Nodes[NodeName]
					CurrentNodeIndex = CurrentNodeIndex + 1

					NodeData.Value = NodeData.Value + NodeData.Bias
					NodeData.WeightedSumVal = NodeData.Value

					if LayerData.ActivationFunction ~= 'None' then
						NodeData.Value = FunctionLib[LayerData.ActivationFunction](NodeData.Value)
					end

					table.insert(OutputVals,NodeData.Value)
				end
			end
		end

		return OutputVals
	end,
	['Backpropagate'] = function(Name,OutputVals,TargetVector,ErrFunction)
		local ANN = NeuralNetworkData[Name]
		local LearningRate = ANN.LearningRate
		local Optimizer = ANN.Optimizer
		local NumLayers = ANN.NumLayers
		local SigmaLoss = 0
		local ClipGradients = ANN['ClipGradients']

		if Optimizer == 'SGD' then
			local LossTable = FunctionLib[ErrFunction](OutputVals,TargetVector)

			for i,v in ipairs(LossTable) do
				SigmaLoss = SigmaLoss + v
			end

			local OutputDerivs = DerivativeFunctionLib[ErrFunction..'Derivative'](OutputVals,TargetVector)

			for LayerIndex = ANN.NumLayers,2,-1 do
				local Layer = 'Layer'..tostring(LayerIndex)

				local c = 0

				for NodeIndex,NodeData in pairs(ANN[Layer].Nodes) do
					c = c + 1
				end

				if LayerIndex == ANN.NumLayers then
					for NodeIndex = 1,c do
						local NodeData = ANN[Layer].Nodes['Node'..tostring(NodeIndex)]

						-- Calculate S values, then backpropagate weights feeding into the layer
						NodeData.S = DerivativeFunctionLib[ANN[Layer].ActivationFunction..'Derivative'](OutputVals[NodeIndex])
						NodeData.S = NodeData.S * OutputDerivs[NodeIndex]
					end
				else
					local A = {{}}

					local NumNodesinNextLayer = 0

					for NodeIndex,NodeData in pairs(ANN['Layer'..tostring(LayerIndex + 1)].Nodes) do
						NumNodesinNextLayer = NumNodesinNextLayer + 1
					end

					for NodeIndex = 1,NumNodesinNextLayer do
						table.insert(A[1],ANN['Layer'..tostring(LayerIndex + 1)].Nodes['Node'..tostring(NodeIndex)].S)
					end

					local B = {}

					local NumNodesinCurLayer = 0

					for NodeIndex,NodeData in pairs(ANN[Layer].Nodes) do
						NumNodesinCurLayer = NumNodesinCurLayer + 1
					end

					for RowIndex = 1,NumNodesinNextLayer do
						table.insert(B,table.create(NumNodesinCurLayer,0))
					end

					for NodeIndex = 1,NumNodesinCurLayer do
						local NodeData = ANN[Layer].Nodes['Node'..tostring(NodeIndex)]

						for ConnectionIndex,ConnectionData in pairs(NodeData.Connections) do
							local RowIndex = tonumber(string.match(ConnectionData.ConnectedNode,'%d+'))
							local ColumnIndex = NodeIndex

							B[RowIndex][ColumnIndex] = ConnectionData.Weight
						end
					end

					local Result = FunctionLib.MatrixMultiplication(A,B)

					for ColumnIndex = 1,NumNodesinCurLayer do
						Result[1][ColumnIndex] = Result[1][ColumnIndex] * DerivativeFunctionLib[ANN[Layer].ActivationFunction..'Derivative'](ANN[Layer].Nodes['Node'..tostring(ColumnIndex)].Value)
					end

					for NodeIndex = 1,NumNodesinCurLayer do
						ANN[Layer].Nodes['Node'..tostring(NodeIndex)].S = Result[1][NodeIndex]
					end
				end

				for Node,NodeData in pairs(ANN['Layer'..tostring(LayerIndex - 1)].Nodes) do
					for ConnectionIndex,ConnectionData in pairs(NodeData.Connections) do
						if ClipGradients then
							ConnectionData.Gradient = math.clamp(NodeData.Value * ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode].S,ANN.ClipGradients[1],ANN.ClipGradients[2])
						else
							ConnectionData.Gradient = NodeData.Value * ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode].S
						end
					end
				end

				for Node,NodeData in pairs(ANN['Layer'..tostring(LayerIndex)].Nodes) do
					if ClipGradients then
						NodeData.Gradient = math.clamp(NodeData.S,ANN.ClipGradients[1],ANN.ClipGradients[2])
					else
						NodeData.Gradient = NodeData.S
					end
				end
			end

			for LayerIndex = 1,ANN.NumLayers do
				local Layer = ANN['Layer'..tostring(LayerIndex)]

				for NodeIndex,NodeData in pairs(Layer.Nodes) do

					if LayerIndex ~= 1 then
						NodeData.Bias = NodeData.Bias - LearningRate * NodeData.Gradient
						NodeData.Gradient = 0
						NodeData.S = 0
					end

					if LayerIndex ~= ANN.NumLayers then
						for ConnectionIndex,ConnectionData in pairs(NodeData.Connections) do
							ConnectionData.Weight = ConnectionData.Weight - LearningRate * ConnectionData.Gradient
							ConnectionData.Gradient = 0
						end
					end
				end
			end
		elseif Optimizer == 'SGD_Momentum' then
			local LossTable = FunctionLib[ErrFunction](OutputVals,TargetVector)

			for i,v in ipairs(LossTable) do
				SigmaLoss = SigmaLoss + v
			end

			local OutputDerivs = DerivativeFunctionLib[ErrFunction..'Derivative'](OutputVals,TargetVector)

			for LayerIndex = ANN.NumLayers,2,-1 do
				local Layer = 'Layer'..tostring(LayerIndex)

				local c = 0

				for NodeIndex,NodeData in pairs(ANN[Layer].Nodes) do
					c = c + 1
				end

				if LayerIndex == ANN.NumLayers then
					for NodeIndex = 1,c do
						local NodeData = ANN[Layer].Nodes['Node'..tostring(NodeIndex)]

						-- Calculate S values, then backpropagate weights feeding into the layer
						NodeData.S = DerivativeFunctionLib[ANN[Layer].ActivationFunction..'Derivative'](OutputVals[NodeIndex])
						NodeData.S = NodeData.S * OutputDerivs[NodeIndex]
					end
				else
					local A = {{}}

					local NumNodesinNextLayer = 0

					for NodeIndex,NodeData in pairs(ANN['Layer'..tostring(LayerIndex + 1)].Nodes) do
						NumNodesinNextLayer = NumNodesinNextLayer + 1
					end

					for NodeIndex = 1,NumNodesinNextLayer do
						table.insert(A[1],ANN['Layer'..tostring(LayerIndex + 1)].Nodes['Node'..tostring(NodeIndex)].S)
					end

					local B = {}

					local NumNodesinCurLayer = 0

					for NodeIndex,NodeData in pairs(ANN[Layer].Nodes) do
						NumNodesinCurLayer = NumNodesinCurLayer + 1
					end

					for RowIndex = 1,NumNodesinNextLayer do
						table.insert(B,table.create(NumNodesinCurLayer,0))
					end

					for NodeIndex = 1,NumNodesinCurLayer do
						local NodeData = ANN[Layer].Nodes['Node'..tostring(NodeIndex)]

						for ConnectionIndex,ConnectionData in pairs(NodeData.Connections) do
							local RowIndex = tonumber(string.match(ConnectionData.ConnectedNode,'%d+'))
							local ColumnIndex = NodeIndex

							B[RowIndex][ColumnIndex] = ConnectionData.Weight
						end
					end

					local Result = FunctionLib.MatrixMultiplication(A,B)

					for ColumnIndex = 1,NumNodesinCurLayer do
						Result[1][ColumnIndex] = Result[1][ColumnIndex] * DerivativeFunctionLib[ANN[Layer].ActivationFunction..'Derivative'](ANN[Layer].Nodes['Node'..tostring(ColumnIndex)].Value)
					end

					for NodeIndex = 1,NumNodesinCurLayer do
						ANN[Layer].Nodes['Node'..tostring(NodeIndex)].S = Result[1][NodeIndex]
					end
				end

				for Node,NodeData in pairs(ANN['Layer'..tostring(LayerIndex - 1)].Nodes) do
					for ConnectionIndex,ConnectionData in pairs(NodeData.Connections) do
						if ClipGradients then
							ConnectionData.Gradient = math.clamp(NodeData.Value * ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode].S,ANN.ClipGradients[1],ANN.ClipGradients[2])
						else
							ConnectionData.Gradient = NodeData.Value * ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode].S
						end
					end
				end

				for Node,NodeData in pairs(ANN['Layer'..tostring(LayerIndex)].Nodes) do
					if ClipGradients then
						NodeData.Gradient = math.clamp(NodeData.S,ANN.ClipGradients[1],ANN.ClipGradients[2])
					else
						NodeData.Gradient = NodeData.S
					end
				end
			end

			for LayerIndex = 1,ANN.NumLayers do
				local Layer = ANN['Layer'..tostring(LayerIndex)]

				for NodeIndex,NodeData in pairs(Layer.Nodes) do

					if LayerIndex ~= 1 then
						local CurVel = ANN.Momentum * NodeData.PrevVelocity + LearningRate * NodeData.Gradient

						NodeData.Bias = NodeData.Bias - CurVel
						NodeData.Gradient = 0
						NodeData.PrevVelocity = CurVel
						NodeData.S = 0
					end

					if LayerIndex ~= ANN.NumLayers then
						for ConnectionIndex,ConnectionData in pairs(NodeData.Connections) do
							local CurVel = ANN.Momentum * ConnectionData.PrevVelocity + LearningRate * ConnectionData.Gradient

							ConnectionData.Weight = ConnectionData.Weight - CurVel
							ConnectionData.PrevVelocity = CurVel
							ConnectionData.Gradient = 0
						end
					end
				end
			end
		end

		return SigmaLoss
	end,
	['ChangeLayerActivationFunction'] = function(Name,LayerName,ActivationFunction)
		NeuralNetworkData[Name][LayerName].ActivationFunction = ActivationFunction

		if NeuralNetworkData[Name][LayerName].LayerType ~= 'Last' then
			NeuralNetworkData[Name][LayerName].LayerType = 'ActivationLayer'
		end
	end,
	['ClipGradients'] = function(Name,min,max)
		NeuralNetworkData[Name].ClipGradients = {min,max}
	end,
	['DecayLearningRate'] = function(Name,DecayRate,EpochNum)
		NeuralNetworkData[Name].LearningRate = (1 / (1 + DecayRate * EpochNum)) * NeuralNetworkData[Name].OriginalLearningRate
	end,
}

local SS = game:GetService('ServerStorage')
local Score = 0
local Alive = true
local Rand = Random.new()
local DatastoreService = game:GetService('DataStoreService')
local NeuralNetworkStore = DatastoreService:GetDataStore('NeuralNetwork1')
local Data
local Epsilon
local Round
local LossData
local RewData
local Epsilon_Decay = 0.995
local CopyWeightsEvery = 16
local Gamma = 0.95
local BatchSize = 64
local MaxMemorySize = 10000

--Default parameters
local lr = 0.001
local NumInputs = 11
local NumHiddenNodes = 13
local NumHiddenLayers = 3
local NumOutputs = 4
local WeightMin = -1
local WeightMax = 1
local HiddenLayerActivationFunction = 'ReLU'
local OutputLayerActivationFunction = 'None'
local Optimizer = 'SGD_Momentum'

local RS = game:GetService('ReplicatedStorage')

game.Players.PlayerAdded:Connect(function(Player)
	Player.Chatted:Connect(function(Message)
		if string.lower(Message) == '!getlossdata' then
			RS.SendData:FireClient(Player,LossData)
		elseif string.lower(Message) == '!getrewarddata' then
			RS.SendData:FireClient(Player,RewData)
		end
	end)
end)

local function UpdateTargetNetwork()
	NeuralNetworkData['TNetwork'] = nil
	
	NeuralNetworkFramework.CreateNN('TNetwork',lr,NumInputs,NumHiddenNodes,NumHiddenLayers,NumOutputs,WeightMin,WeightMax,HiddenLayerActivationFunction,OutputLayerActivationFunction,Optimizer,{Momentum=0.99})
	
	for i = 1,NeuralNetworkData.QNetwork.NumLayers do
		local Layer = 'Layer'..tostring(i)
		local LayerData = NeuralNetworkData.QNetwork[Layer]
		
		for j,v in pairs(LayerData.Nodes) do
			NeuralNetworkData.TNetwork[Layer].Nodes[j].Bias = v.Bias
			
			if i ~= NeuralNetworkData.QNetwork.NumLayers then
				for k,v2 in pairs(v.Connections) do
					local ConnectionData = NeuralNetworkData.TNetwork[Layer].Nodes[j].Connections[k]
					
					ConnectionData.Weight = v2.Weight
					ConnectionData.ConnectedLayer = v2.ConnectedLayer
					ConnectionData.ConnectedNode = v2.ConnectedNode
					-- TargetNetwork --> Layer 1 --> Nodes --> Node j --> Connections --> Connection k
				end
			end
		end
	end
end

local function CreateNetworks()
	NeuralNetworkFramework.CreateNN('QNetwork',lr,NumInputs,NumHiddenNodes,NumHiddenLayers,NumOutputs,WeightMin,WeightMax,HiddenLayerActivationFunction,OutputLayerActivationFunction,Optimizer,{Momentum=0.99})
	NeuralNetworkFramework.ClipGradients('QNetwork',-1,1)
	
	UpdateTargetNetwork()
	
	Epsilon = 1--0.05
	Round = 1
	LossData = {Loss = {}}
	RewData = {AvgReward = {}} -- Will eventually look like RewData = {Reward = {1,2,3,4,2,5,2,7,...}} and LossData = {Loss = {1,25,2,3,1,0.5,0.8,0.3,...}}
end

local function Save()
	local DataToSave = {Epsilon = Epsilon,Round = Round,LossData = LossData,RewData = RewData,QNetwork = NeuralNetworkData.QNetwork,TNetwork = NeuralNetworkData.TNetwork}
	
	local Success,Err = pcall(function()
		NeuralNetworkStore:SetAsync('joe1234',DataToSave)
	end)
	
	if not Success then
		warn(Err)
	end
end

local Success,Err = pcall(function()
	Data = NeuralNetworkStore:GetAsync('joe1234')
end)

if not Success then
	warn(Err)
	CreateNetworks()
else
	NeuralNetworkData.QNetwork = Data.QNetwork
	NeuralNetworkData.TNetwork = Data.TNetwork
	Epsilon = Data.Epsilon
	Round = Data.Round
	LossData = Data.LossData
	RewData = Data.RewData
end

if Data == nil then
	CreateNetworks()
end

local function AddApple()
	local Apple = SS.Apple:Clone()
	local PossiblePositions = {}
	local SnakePositions = {}
	
	for i = 1,Score + 1 do
		table.insert(SnakePositions,workspace.Snake[tostring(i)].Position)
	end
	
	table.insert(SnakePositions,workspace.Snake.SnakeHead.Position)
	
	for i = 0,22.5 - 5.5 do
		local x = 5.5 + i
		
		for j = -47.5 - -29.5,0 do
			local z = -47.5 - j
			
			local P = Vector3.new(x,0.5,z)
			
			if table.find(SnakePositions,P) ~= nil then
				continue
			end
			
			table.insert(PossiblePositions,P)
		end
	end
	
	local ChosenPos = PossiblePositions[math.random(1,#PossiblePositions)]

	Apple.Position = ChosenPos
	Apple.Parent = workspace
end

local function StartGame()
	local SnakeHead = SS.SnakeHead:Clone()
	local SnakeTail = SS.SnakeTail:Clone()
	
	local X = math.round(Rand:NextNumber(5.5,22.5) - 0.5) + 0.5
	local Y = 0.5
	local Z = math.round(Rand:NextNumber(-47.5,-29.5) - 0.5) + 0.5
	
	SnakeHead.CFrame = CFrame.new(X,Y,Z) * CFrame.Angles(0,math.rad(90),0)--workspace.Spawn.CFrame
	SnakeHead.Parent = workspace.Snake
	
	SnakeTail.CFrame = SnakeHead.CFrame - SnakeHead.CFrame.LookVector
	SnakeTail.Name = '1'
	SnakeTail.Parent = workspace.Snake
end

local Reward = 0

local function AddTail()
	local SnakeTail = SS.SnakeTail:Clone()

	SnakeTail.CFrame = workspace.Snake[tostring(Score + 1)].CFrame - workspace.Snake[tostring(Score + 1)].CFrame.LookVector
	Score += 1
	Reward = 10
	SnakeTail.Name = tostring(Score + 1)
	SnakeTail.Parent = workspace.Snake
end

StartGame()

local function MoveSnake()
	local OldPositions = {}
	
	for i = 1,Score + 1 do
		local v = workspace.Snake[tostring(i)]
		
		table.insert(OldPositions,v.Position)
	end
	
	local NewPositions = {}
	
	for i = 1,Score + 1 do
		local v = workspace.Snake[tostring(i)]
		
		if i == 1 then
			table.insert(NewPositions,workspace.Snake.SnakeHead.CFrame)
		else
			table.insert(NewPositions,workspace.Snake[tostring(i - 1)].CFrame)
		end
	end
	
	for i = 1,Score + 1 do
		local v = workspace.Snake[tostring(i)]
		
		v.CFrame = NewPositions[i]
	end
	
	workspace.Snake.SnakeHead.CFrame += workspace.Snake.SnakeHead.CFrame.LookVector
	
	local List = workspace.Snake.SnakeHead:GetTouchingParts()
	
	for i,v in pairs(List) do
		if v.Parent == workspace.Snake or v.Name == 'Border' then
			Alive = false
			workspace.Snake.SnakeHead.CFrame -= workspace.Snake.SnakeHead.CFrame.LookVector
			
			for i = 1,Score + 1 do
				local v = workspace.Snake[tostring(i)]
				
				v.Position = OldPositions[i]
			end
			
			Reward = -10 --Crashed
			break
		elseif v.Name == 'Apple' then
			AddTail()
			v:Destroy()
			
			AddApple()
			break
		end
	end
end

local function UpArrow()
	if workspace.Snake.SnakeHead.Orientation ~= Vector3.new(0,0,0) then
		workspace.Snake.SnakeHead.Orientation = Vector3.new(0,180,0)
	end
end

local function LeftArrow()
	if workspace.Snake.SnakeHead.Orientation ~= Vector3.new(0,90,0) then
		workspace.Snake.SnakeHead.Orientation = Vector3.new(0,-90,0)
	end
end

local function RightArrow()
	if workspace.Snake.SnakeHead.Orientation ~= Vector3.new(0,-90,0) then
		workspace.Snake.SnakeHead.Orientation = Vector3.new(0,90,0)
	end
end

local function DownArrow()
	if workspace.Snake.SnakeHead.Orientation ~= Vector3.new(0,-180,0) then
		workspace.Snake.SnakeHead.Orientation = Vector3.new(0,0,0)
	end
end

local function CalculateInputs()
	local Head = workspace.Snake.SnakeHead
	local Apple = workspace.Apple
	local Return = {}

	if Apple.Position.Z > Head.Position.Z then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	if Apple.Position.Z < Head.Position.Z then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	if Apple.Position.X > Head.Position.X then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	if Apple.Position.X < Head.Position.X then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	local Params = RaycastParams.new()

	Params.FilterDescendantsInstances = {Apple,workspace.Spawn}
	Params.FilterType = Enum.RaycastFilterType.Blacklist

	local RayOrigin = Head.Position
	local RayDirection = Head.Position + Head.CFrame.LookVector - Head.Position

	local Result = workspace:Raycast(RayOrigin,RayDirection,Params)

	if Result ~= nil then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	RayDirection = Head.Position + Head.CFrame.RightVector * -1 - Head.Position

	Result = workspace:Raycast(RayOrigin,RayDirection,Params)

	if Result ~= nil then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	RayDirection = Head.Position + Head.CFrame.RightVector - Head.Position

	Result = workspace:Raycast(RayOrigin,RayDirection,Params)

	if Result ~= nil then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	if Head.Orientation == Vector3.new(0,-180,0) then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	if Head.Orientation == Vector3.new(0,0,0) then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	if Head.Orientation == Vector3.new(0,-90,0) then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	if Head.Orientation == Vector3.new(0,90,0) then
		table.insert(Return,1)
	else
		table.insert(Return,0)
	end

	return Return
end

AddApple()

local function Shuffle(t)
	local j, temp

	for i = #t, 1, -1 do
		j = math.random(i)
		temp = t[i]
		t[i] = t[j]
		t[j] = temp
	end

	return t
end

local ExperienceReplayTable = {} --{State_s,Action_a,Reward,Next_s}
local PrevState
local PrevAction
local PrevReward

local function FillExperienceReplayTable(AmtWait,Debug)
	PrevState = nil
	PrevAction = nil
	PrevReward = nil
	
	local EpData = {}
	
	repeat
		if AmtWait > 0 then
			task.wait(AmtWait)
		end

		Reward = 0

		local Action
		local Inputs = CalculateInputs()

		local Probability = Rand:NextNumber()

		if Probability <= Epsilon then
			Action = math.random(1,4)
		else
			Action = math.random(1,4)
			
			local OutputQValues = NeuralNetworkFramework.CalculateForwardPass('QNetwork',Inputs)

			for i,v in pairs(OutputQValues) do
				if v > OutputQValues[Action] then
					Action = i
				end

				if math.abs(v) > 1000 then
					warn('smth wrong')
					warn(OutputQValues)
				end
			end

			if Debug then
				warn('Chosen Action: '..Action)
			end
		end

		PrevState = Inputs
		PrevAction = Action

		local OldDist = (workspace.Snake.SnakeHead.Position - workspace.Apple.Position).Magnitude

		if Action == 1 then
			UpArrow()
		elseif Action == 2 then
			DownArrow()
		elseif Action == 3 then
			RightArrow()
		else
			LeftArrow()
		end

		MoveSnake()

		if Reward == 0 then
			local NewDist = (workspace.Snake.SnakeHead.Position - workspace.Apple.Position).Magnitude

			if NewDist > OldDist then
				Reward = 1
			else
				Reward = -1
			end
		end

		PrevReward = Reward
		
		if Alive then
			table.insert(ExperienceReplayTable,{Cur_s = PrevState,Cur_a = PrevAction,Reward = PrevReward,Next_s = CalculateInputs(),TerminalState = false})
		else
			table.insert(ExperienceReplayTable,{Cur_s = PrevState,Cur_a = PrevAction,Reward = PrevReward,Next_s = CalculateInputs(),TerminalState = false})
		end
	until not Alive

	for i,v in pairs(workspace.Snake:GetChildren()) do
		v:Destroy()
	end

	StartGame()
	Score = 0
	workspace.Apple:Destroy()
	AddApple()

	Round += 1
	Alive = true
end

--[[task.wait(5)
FillExperienceReplayTable(0.1,false)

print(ExperienceReplayTable)

task.wait(5)

local Count = 0
local Clock = os.clock()
local i = 1

for a,b in pairs(ExperienceReplayTable) do
	print('Episode : '..a)
	print(b)
	local Loss = 0
	local AvgReward = 0

	for k,v in pairs(b) do
		local State = v.Cur_s
		local ActionTaken = v.Cur_a
		local Rew = v.Reward
		local Terminal = v.TerminalState
		local NextState
		local Q_Target
		
		print(State)
		print(ActionTaken)
		print(Rew)
		print(Terminal)

		if not Terminal then
			NextState = v.Next_s
			print(NextState)

			local TargetQValues = NeuralNetworkFramework.CalculateForwardPass('TNetwork',NextState)
			local QValues = NeuralNetworkFramework.CalculateForwardPass('QNetwork',NextState)
			local HighestIndex = math.random(1,#QValues)
			
			local Check = false

			for j,v2 in pairs(QValues) do
				if v2 > QValues[HighestIndex] then
					HighestIndex = j
				end
				
				if v2 ~= TargetQValues[j] then
					print('Found difference between TNetwork and QNetwork!')
				end
			end
			
			print(TargetQValues)
			print(QValues)
			print(HighestIndex)

			Q_Target = Rew + Gamma * TargetQValues[HighestIndex]
		else
			Q_Target = Rew
		end

		local Outputs = NeuralNetworkFramework.CalculateForwardPass('QNetwork',State)
		local TargetValues = {}

		for j,v2 in pairs(Outputs) do
			if j ~= ActionTaken then
				table.insert(TargetValues,v2)
			else
				table.insert(TargetValues,Q_Target)
			end
		end
		
		local Error = NeuralNetworkFramework.Backpropagate('QNetwork',Outputs,TargetValues,'MeanSquaredError')
		
		Loss += Error
		AvgReward += Rew

		Count += 1

		if Count == CopyWeightsEvery then
			UpdateTargetNetwork()
			Count = 0
		end

		if i % 100 == 0 then
			task.wait()
		end

		i += 1
	end

	Loss /= #b
	AvgReward /= #b

	table.insert(LossData.Loss,Loss)
	table.insert(RewData.AvgReward,AvgReward)
	break
end

print('Done!')--]]

local function Episode(AmtWait,Debug)
	PrevState = nil
	PrevAction = nil
	PrevReward = nil
	
	local AvgReward = 0
	local NumIterations = 0
	local Loss = 0
	
	repeat
		if AmtWait > 0 then
			task.wait(AmtWait)
		end

		Reward = 0

		local Action
		local Inputs = CalculateInputs()

		local Probability = Rand:NextNumber()

		if Probability <= Epsilon then
			Action = math.random(1,4)
		else
			Action = math.random(1,4)
			
			local OutputQValues = NeuralNetworkFramework.CalculateForwardPass('QNetwork',Inputs)

			if Debug then print(table.concat(OutputQValues,', ')) end

			for i,v in pairs(OutputQValues) do
				if v > OutputQValues[Action] then
					Action = i
				end

				if math.abs(v) > 1000 then
					warn('smth wrong')
					warn(OutputQValues)
				end
			end
		end

		PrevState = Inputs
		PrevAction = Action
		
		local OldDist = (workspace.Snake.SnakeHead.Position - workspace.Apple.Position).Magnitude
		
		if Action == 1 then
			UpArrow()
		elseif Action == 2 then
			DownArrow()
		elseif Action == 3 then
			RightArrow()
		else
			LeftArrow()
		end

		MoveSnake()
		
		if Reward == 0 then
			local NewDist = (workspace.Snake.SnakeHead.Position - workspace.Apple.Position).Magnitude
			
			if NewDist > OldDist then
				Reward = 1
			else
				Reward = -1
			end
		end

		PrevReward = Reward
		
		if Alive then
			table.insert(ExperienceReplayTable,{Cur_s = PrevState,Cur_a = PrevAction,Reward = PrevReward,Next_s = CalculateInputs(),TerminalState = false})
		else
			table.insert(ExperienceReplayTable,{Cur_s = PrevState,Cur_a = PrevAction,Reward = PrevReward,TerminalState = true})
		end
		
		if #ExperienceReplayTable > MaxMemorySize then
			table.remove(ExperienceReplayTable,1)
		end
		
		AvgReward += Reward
		NumIterations += 1
	until not Alive
	
	for k = 1,1 do
		local Batch = {}
		local Count = 0
		local Clock = os.clock()
		
		for i = 1,BatchSize do
			table.insert(Batch,ExperienceReplayTable[math.random(#ExperienceReplayTable)])
		end
		
		Shuffle(Batch)
		
		for i,v in pairs(Batch) do
			local State = v.Cur_s
			local ActionTaken = v.Cur_a
			local Rew = v.Reward
			local Terminal = v.TerminalState
			local NextState
			local Q_Target

			if not Terminal then
				NextState = v.Next_s

				local TargetQValues = NeuralNetworkFramework.CalculateForwardPass('TNetwork',NextState)
				local QValues = NeuralNetworkFramework.CalculateForwardPass('QNetwork',NextState)
				local HighestIndex = math.random(1,#QValues)

				for j,v2 in pairs(QValues) do
					if v2 > QValues[HighestIndex] then
						HighestIndex = j
					end
				end

				Q_Target = Rew + Gamma * TargetQValues[HighestIndex]
			else
				Q_Target = Rew
			end
			
			local Outputs = NeuralNetworkFramework.CalculateForwardPass('QNetwork',State)
			local TargetValues = {}

			for j,v2 in pairs(Outputs) do
				if j ~= ActionTaken then
					table.insert(TargetValues,v2)
				else
					table.insert(TargetValues,Q_Target)
				end
			end
			
			if State[5] == 1 then
				--[[print(Outputs)
				print(ActionTaken)
				print(TargetValues)==]]
			end
			
			local Error = NeuralNetworkFramework.Backpropagate('QNetwork',Outputs,TargetValues,'MeanSquaredError')
			
			Loss += Error
			
			Count += 1

			if Count == CopyWeightsEvery then
				UpdateTargetNetwork()
			end

			if Clock - os.clock() >= 0.1 then
				task.wait()
				Clock = os.clock()
			end
		end
	end
	
	AvgReward /= NumIterations
	Loss /= BatchSize

	for i,v in pairs(workspace.Snake:GetChildren()) do
		v:Destroy()
	end

	StartGame()
	Score = 0
	workspace.Apple:Destroy()
	AddApple()

	Round += 1
	Alive = true

	if Epsilon > 0.05 then
		Epsilon *= Epsilon_Decay
	end
	
	table.insert(RewData.AvgReward,AvgReward)
	table.insert(LossData.Loss,Loss)
end

local c = os.clock()

--[[local NumDataCollectingEpisodes = BatchSize--3200

for i = 1,NumDataCollectingEpisodes do
	FillExperienceReplayTable(0,false)
	
	if os.clock() - c >= 0.01 then
		task.wait()
		c = os.clock()
		print(math.round(i / NumDataCollectingEpisodes * 100)..'%')
	end
end--]]

--[[print('Going to peform gradient descent in 5 seconds!')

task.wait(5)

print('Start!')

task.wait()

local Count = 0
local Clock = os.clock()
local i = 1

local Loss = 0
local AvgReward = 0

for h = 1,NumDataCollectingEpisodes do
	local Batch = {}
	
	for j = 1,BatchSize do
		table.insert(Batch,ExperienceReplayTable[math.random(1,#ExperienceReplayTable)])
	end
	
	Shuffle(Batch)
	
	for k,v in pairs(Batch) do
		local State = v.Cur_s
		local ActionTaken = v.Cur_a
		local Rew = v.Reward
		local Terminal = v.TerminalState
		local NextState
		local Q_Target

		if not Terminal then
			NextState = v.Next_s

			local TargetQValues = NeuralNetworkFramework.CalculateForwardPass('TNetwork',NextState)
			local QValues = NeuralNetworkFramework.CalculateForwardPass('QNetwork',NextState)
			local HighestIndex = math.random(1,#QValues)

			for j,v2 in pairs(QValues) do
				if v2 > QValues[HighestIndex] then
					HighestIndex = j
				end
			end

			Q_Target = Rew + Gamma * TargetQValues[HighestIndex]
		else
			Q_Target = Rew
		end

		local Outputs = NeuralNetworkFramework.CalculateForwardPass('QNetwork',State)
		local TargetValues = {}

		for j,v2 in pairs(Outputs) do
			if j ~= ActionTaken then
				table.insert(TargetValues,v2)
			else
				table.insert(TargetValues,Q_Target)
			end
		end

		local Error = NeuralNetworkFramework.Backpropagate('QNetwork',Outputs,TargetValues,'MeanSquaredError')
		
		Loss += Error
		AvgReward += Rew
		
		Count += 1

		if Count == CopyWeightsEvery then
			UpdateTargetNetwork()
			Count = 0
		end
		
		if os.clock() - Clock >= 0.1 then
			print(math.round((h / NumDataCollectingEpisodes) * 100)..'% done')
			task.wait()
			Clock = os.clock()
		end
		
		i += 1
	end
	
	Loss /= #Batch
	AvgReward /= #Batch

	table.insert(LossData.Loss,Loss)
	table.insert(RewData.AvgReward,AvgReward)
end

print('Done!')

Epsilon = 0--]]

--Episode(0.1,true)

c = os.clock()

repeat
	print('Round: '..Round)
	print('Epsilon: '..Epsilon)
	
	Episode(0,false)
	
	if os.clock() - c >= 0.1 then
		task.wait()
		c = os.clock()
		--print(math.round((Round / 2000) * 100)..'% done')
		print(math.round(math.clamp((0.05 / Epsilon),0,1) * 100)..'% done')
	end
until Epsilon <= 0.05--recomment later

--Episode(0.1,true)

while true do
	print('Round: '..Round)
	print('Epsilon: '..Epsilon)

	Episode(0.001,false)
end

for i,v in pairs(RewData.AvgReward) do
	if v > 2 then
		return true
	end
end

return false
