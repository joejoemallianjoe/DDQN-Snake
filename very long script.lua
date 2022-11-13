math.randomseed(os.clock() - os.time())

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
	['TanHDerivative'] = function(y)
		local function TanH(x)
			return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
		end

		return 1 - TanH(y)^2
	end
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
		return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
	end,
	['DotProduct'] = function(Matrix1,Matrix2)
		local ReturnMatrix = {}

		for i,v in pairs(Matrix1) do
			table.insert(ReturnMatrix,v * Matrix2[i])
		end

		return ReturnMatrix
	end,
	['MeanSquaredError'] = function(OutputVector,TargetVector)
		local ReturnVector = {}

		for i,v in ipairs(TargetVector) do
			local o = OutputVector[i]
			table.insert(ReturnVector,0.5 * (v - o)^2)
		end

		return ReturnVector
	end
}

local function CalculateDerivativePath(Name,NodeData,ConnectionData,EndNodeIndex,OutputVector,TargetVector,ErrFunction,IsBias)
	local function CalculateDerivative(ND,ConnectData,WRT)
		if WRT == 'Weight' then
			return ND.Value
		elseif WRT == 'WS' then
			return ConnectData.Weight
		else
			--WRT == 'ActivationFunction'
			local ConnectedNodeData = NeuralNetworkData[Name][ConnectData.ConnectedLayer].Nodes[ConnectData.ConnectedNode]
			local deriv = ConnectedNodeData.WeightedSumVal
			local ConnectedNodeActivationFunction = NeuralNetworkData[Name][ConnectData.ConnectedLayer].ActivationFunction
			deriv = DerivativeFunctionLib[ConnectedNodeActivationFunction..'Derivative'](deriv)
			return deriv
		end
	end

	local function ListCombinations(NumLayers,StartLayer,NumHiddenNodes,EndNodeInd)
		local NumDigits = NumLayers - StartLayer
		local NumForiLoops = NumDigits - 1
		local VariablesTable = {}
		local Count = 0
		local t = {}

		if NumDigits == 1 then				
			table.insert(t,{EndNodeInd})
			return t
		end

		local function RecursivelyCreateForiLoop(Amount,MaxIterations)
			Count = Count + 1

			local CurrentForLoopIndex = Count

			for i = 1,MaxIterations do
				VariablesTable['Var'..tostring(CurrentForLoopIndex)] = i

				if Count ~= Amount then
					RecursivelyCreateForiLoop(Amount,MaxIterations)
				else
					local TableToInsert = {}

					for j = 1,NumForiLoops do
						table.insert(TableToInsert,VariablesTable['Var'..tostring(j)])
					end

					table.insert(TableToInsert,EndNodeInd)

					table.insert(t,TableToInsert)
				end
			end

			if Count == Amount then
				Count = Count - 1
			end
		end

		RecursivelyCreateForiLoop(NumForiLoops,NumHiddenNodes)
		return t
	end

	local Num = string.gsub(ConnectionData.ConnectedLayer,'%D+','')
	Num = tonumber(Num)

	if Num == NeuralNetworkData[Name].NumLayers then
		local ChainRuleTable = {}

		if not IsBias then
			table.insert(ChainRuleTable,CalculateDerivative(NodeData,ConnectionData,'Weight'))
		end

		local Ref = NeuralNetworkData[Name]

		if Ref[ConnectionData.ConnectedLayer].ActivationFunction ~= 'None' then
			table.insert(ChainRuleTable,CalculateDerivative(NodeData,ConnectionData,'ActivationFunction'))
		end

		table.insert(ChainRuleTable,DerivativeFunctionLib[ErrFunction..'Derivative'](OutputVector,TargetVector)[EndNodeIndex])

		local PartialDerivative = 1

		for j,v in ipairs(ChainRuleTable) do
			PartialDerivative = PartialDerivative * v
		end

		return PartialDerivative
	end

	local NumberOfNodes = 0

	for i,v in ipairs(NeuralNetworkData[Name][ConnectionData.ConnectedLayer].Nodes) do
		NumberOfNodes = NumberOfNodes + 1
	end

	local Combinations = ListCombinations(NeuralNetworkData[Name].NumLayers,Num,NumberOfNodes,EndNodeIndex)
	local CombinedPartialDerivatives = 0

	for i,v in ipairs(Combinations) do
		local Ref = NeuralNetworkData[Name]
		local ChainRuleTable = {}
		--warn('Index: '..i)
		--warn('Combination: '..table.concat(v,', '))
		if not IsBias then
			table.insert(ChainRuleTable,CalculateDerivative(NodeData,ConnectionData,'Weight'))
		end

		if Ref[ConnectionData.ConnectedLayer].ActivationFunction ~= 'None' then
			table.insert(ChainRuleTable,CalculateDerivative(NodeData,ConnectionData,'ActivationFunction'))
		end

		local CurrentNodeIndex = Ref[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode]
		-- Calculate derivatives for next layer
		for j = 1 + Num,#v + Num do
			local ConvertedCombinationIndex = j - Num
			local LayerNode = 'Node'..tostring(v[ConvertedCombinationIndex])
			local CurNodeIndexConnections = CurrentNodeIndex.Connections
			local CurNodeIndexConnectionData

			for k,v1 in ipairs(CurNodeIndexConnections) do
				if v1.ConnectedNode == LayerNode then
					CurNodeIndexConnectionData = v1
				end
			end
			--Find connection between this node and start node
			table.insert(ChainRuleTable,CalculateDerivative(CurrentNodeIndex,CurNodeIndexConnectionData,'WS'))

			if Ref[CurNodeIndexConnectionData.ConnectedLayer].ActivationFunction ~= 'None' then
				table.insert(ChainRuleTable,CalculateDerivative(CurrentNodeIndex,CurNodeIndexConnectionData,'ActivationFunction'))
			end

			CurrentNodeIndex = Ref['Layer'..tostring(j)].Nodes[LayerNode]
		end

		table.insert(ChainRuleTable,DerivativeFunctionLib[ErrFunction..'Derivative'](OutputVector,TargetVector)[EndNodeIndex])

		local PartialDerivative = 1

		for j,v in ipairs(ChainRuleTable) do
			PartialDerivative = PartialDerivative * v
		end

		CombinedPartialDerivatives = CombinedPartialDerivatives + PartialDerivative
	end

	return CombinedPartialDerivatives
end

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
					['PrevVelocity'] = 0
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
				['PrevVelocity'] = 0
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

		if Optimizer == 'SGD' then
			local LastLayer = ANN['Layer'..tostring(NumLayers)]
			local LossTable = FunctionLib[ErrFunction](OutputVals,TargetVector)

			for i,v in ipairs(LossTable) do
				SigmaLoss = SigmaLoss + v
			end
			local t1 = os.clock()
			local Gradients = {}
			local test

			for i = 1,NumLayers - 1 do -- Output layer should be excluded
				local CurLayer = 'Layer'..tostring(i)

				for j,NodeData in pairs(ANN[CurLayer].Nodes) do
					for k,ConnectionData in ipairs(NodeData.Connections) do
						local t2 = os.clock()
						local PartialDerivatives = {}

						for l = 1,#TargetVector do
							table.insert(PartialDerivatives,CalculateDerivativePath(Name,NodeData,ConnectionData,l,OutputVals,TargetVector,ErrFunction,false))
						end

						local CombinedPartialDerivatives = 0

						for l,v in ipairs(PartialDerivatives) do
							CombinedPartialDerivatives = CombinedPartialDerivatives + v
						end

						if ANN['ClipGradients'] ~= nil then
							CombinedPartialDerivatives = math.clamp(CombinedPartialDerivatives,ANN.ClipGradients[1],ANN.ClipGradients[2])
						end

						table.insert(Gradients,{ConnectionData,'Weight',ConnectionData.Weight - LearningRate * CombinedPartialDerivatives})

						PartialDerivatives = {}
						CombinedPartialDerivatives = 0

						--Update biases for nodes in next layer
						local Ref = ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode]

						if not Ref.BiasUpdated then
							for l = 1,#TargetVector do
								table.insert(PartialDerivatives,CalculateDerivativePath(Name,NodeData,ConnectionData,l,OutputVals,TargetVector,ErrFunction,true))
							end

							for l,v in ipairs(PartialDerivatives) do
								CombinedPartialDerivatives = CombinedPartialDerivatives + v
							end

							if ANN['ClipGradients'] ~= nil then
								CombinedPartialDerivatives = math.clamp(CombinedPartialDerivatives,ANN.ClipGradients[1],ANN.ClipGradients[2])
							end

							table.insert(Gradients,{Ref,'Bias',Ref.Bias - LearningRate * CombinedPartialDerivatives})

							Ref.BiasUpdated = true
						end
					end
				end
			end

			for i,v in pairs(Gradients) do
				v[1][v[2]] = v[3]
			end
		elseif Optimizer == 'Minibatch' then
			local LastLayer = ANN['Layer'..tostring(NumLayers)]
			local LossTable = FunctionLib[ErrFunction](OutputVals,TargetVector)
			local BatchSize = ANN.BatchSize

			ANN.NumGradientSteps = ANN.NumGradientSteps + 1

			for i,v in ipairs(LossTable) do
				SigmaLoss = SigmaLoss + v
			end

			local t1 = os.clock()
			local Gradients = {}

			for i = 1,NumLayers - 1 do -- Output layer should be excluded
				local CurLayer = 'Layer'..tostring(i)

				for j,NodeData in pairs(ANN[CurLayer].Nodes) do
					for k,ConnectionData in ipairs(NodeData.Connections) do
						local t2 = os.clock()
						local PartialDerivatives = {}

						for l = 1,#TargetVector do
							table.insert(PartialDerivatives,CalculateDerivativePath(Name,NodeData,ConnectionData,l,OutputVals,TargetVector,ErrFunction,false))
						end

						local CombinedPartialDerivatives = 0

						for l,v in ipairs(PartialDerivatives) do
							CombinedPartialDerivatives = CombinedPartialDerivatives + v
						end

						if ANN['ClipGradients'] ~= nil then
							CombinedPartialDerivatives = math.clamp(CombinedPartialDerivatives,ANN.ClipGradients[1],ANN.ClipGradients[2])
						end

						ConnectionData.Gradient = ConnectionData.Gradient + CombinedPartialDerivatives -- Not learning rate * gradient yet

						PartialDerivatives = {}
						CombinedPartialDerivatives = 0

						local Ref = ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode]

						if not Ref.BiasUpdated then
							for l = 1,#TargetVector do
								table.insert(PartialDerivatives,CalculateDerivativePath(Name,NodeData,ConnectionData,l,OutputVals,TargetVector,ErrFunction,true))
							end

							for l,v in ipairs(PartialDerivatives) do
								CombinedPartialDerivatives = CombinedPartialDerivatives + v
							end

							if ANN['ClipGradients'] ~= nil then
								CombinedPartialDerivatives = math.clamp(CombinedPartialDerivatives,ANN.ClipGradients[1],ANN.ClipGradients[2])
							end

							Ref.Gradient = Ref.Gradient + CombinedPartialDerivatives

							Ref.BiasUpdated = true
						end
					end
				end
			end

			if ANN.NumGradientSteps == BatchSize then
				--Average the gradients out
				for i = 1,NumLayers - 1 do
					local CurLayer = 'Layer'..tostring(i)

					for j, NodeData in pairs(ANN[CurLayer].Nodes) do
						for k,ConnectionData in ipairs(NodeData.Connections) do
							local t2 = os.clock()
							ConnectionData.Weight = ConnectionData.Weight - LearningRate * (ConnectionData.Gradient / BatchSize)

							ConnectionData.Gradient = 0

							local Ref = ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode]

							Ref.Bias = Ref.Bias - LearningRate * (Ref.Gradient / BatchSize)
							Ref.Gradient = 0
						end
					end
				end

				ANN.NumGradientSteps = 0
			end
		elseif Optimizer == 'SGD_Momentum' then
			local LastLayer = ANN['Layer'..tostring(NumLayers)]
			local LossTable = FunctionLib[ErrFunction](OutputVals,TargetVector)

			for i,v in ipairs(LossTable) do
				SigmaLoss = SigmaLoss + v
			end
			local t1 = os.clock()
			local Gradients = {}
			local test

			for i = 1,NumLayers - 1 do -- Output layer should be excluded
				local CurLayer = 'Layer'..tostring(i)

				for j,NodeData in pairs(ANN[CurLayer].Nodes) do
					for k,ConnectionData in ipairs(NodeData.Connections) do
						local t2 = os.clock()
						local PartialDerivatives = {}

						for l = 1,#TargetVector do
							table.insert(PartialDerivatives,CalculateDerivativePath(Name,NodeData,ConnectionData,l,OutputVals,TargetVector,ErrFunction,false))
						end

						local CombinedPartialDerivatives = 0

						for l,v in ipairs(PartialDerivatives) do
							CombinedPartialDerivatives = CombinedPartialDerivatives + v
						end

						if ANN['ClipGradients'] ~= nil then
							CombinedPartialDerivatives = math.clamp(CombinedPartialDerivatives,ANN.ClipGradients[1],ANN.ClipGradients[2])
						end

						local CurVelocity = ConnectionData.PrevVelocity * ANN.Momentum + LearningRate * CombinedPartialDerivatives

						table.insert(Gradients,{ConnectionData,'Weight',ConnectionData.Weight - CurVelocity})

						ConnectionData.PrevVelocity = CurVelocity

						PartialDerivatives = {}
						CombinedPartialDerivatives = 0

						--Update biases for nodes in next layer
						local Ref = ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode]

						if not Ref.BiasUpdated then
							for l = 1,#TargetVector do
								table.insert(PartialDerivatives,CalculateDerivativePath(Name,NodeData,ConnectionData,l,OutputVals,TargetVector,ErrFunction,true))
							end

							for l,v in ipairs(PartialDerivatives) do
								CombinedPartialDerivatives = CombinedPartialDerivatives + v
							end

							if ANN['ClipGradients'] ~= nil then
								CombinedPartialDerivatives = math.clamp(CombinedPartialDerivatives,ANN.ClipGradients[1],ANN.ClipGradients[2])
							end

							CurVelocity = Ref.PrevVelocity * ANN.Momentum + LearningRate * CombinedPartialDerivatives

							table.insert(Gradients,{Ref,'Bias',Ref.Bias - CurVelocity})

							Ref.PrevVelocity = CurVelocity

							Ref.BiasUpdated = true
						end
					end
				end
			end

			for i,v in pairs(Gradients) do
				v[1][v[2]] = v[3]
			end
		elseif Optimizer == 'Minibatch_Momentum' then
			local LastLayer = ANN['Layer'..tostring(NumLayers)]
			local LossTable = FunctionLib[ErrFunction](OutputVals,TargetVector)
			local BatchSize = ANN.BatchSize

			ANN.NumGradientSteps = ANN.NumGradientSteps + 1

			for i,v in ipairs(LossTable) do
				SigmaLoss = SigmaLoss + v
			end

			local t1 = os.clock()
			local Gradients = {}

			for i = 1,NumLayers - 1 do -- Output layer should be excluded
				local CurLayer = 'Layer'..tostring(i)

				for j,NodeData in pairs(ANN[CurLayer].Nodes) do
					for k,ConnectionData in ipairs(NodeData.Connections) do
						local t2 = os.clock()
						local PartialDerivatives = {}

						for l = 1,#TargetVector do
							table.insert(PartialDerivatives,CalculateDerivativePath(Name,NodeData,ConnectionData,l,OutputVals,TargetVector,ErrFunction,false))
						end

						local CombinedPartialDerivatives = 0

						for l,v in ipairs(PartialDerivatives) do
							CombinedPartialDerivatives = CombinedPartialDerivatives + v
						end

						if ANN['ClipGradients'] ~= nil then
							CombinedPartialDerivatives = math.clamp(CombinedPartialDerivatives,ANN.ClipGradients[1],ANN.ClipGradients[2])
						end

						ConnectionData.Gradient = ConnectionData.Gradient + CombinedPartialDerivatives -- Not learning rate * gradient yet

						PartialDerivatives = {}
						CombinedPartialDerivatives = 0

						local Ref = ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode]

						if not Ref.BiasUpdated then
							for l = 1,#TargetVector do
								table.insert(PartialDerivatives,CalculateDerivativePath(Name,NodeData,ConnectionData,l,OutputVals,TargetVector,ErrFunction,true))
							end

							for l,v in ipairs(PartialDerivatives) do
								CombinedPartialDerivatives = CombinedPartialDerivatives + v
							end

							if ANN['ClipGradients'] ~= nil then
								CombinedPartialDerivatives = math.clamp(CombinedPartialDerivatives,ANN.ClipGradients[1],ANN.ClipGradients[2])
							end

							Ref.Gradient = Ref.Gradient + CombinedPartialDerivatives

							Ref.BiasUpdated = true
						end
					end
				end
			end

			if ANN.NumGradientSteps == BatchSize then
				--Average the gradients out
				for i = 1,NumLayers - 1 do
					local CurLayer = 'Layer'..tostring(i)

					for j, NodeData in pairs(ANN[CurLayer].Nodes) do
						for k,ConnectionData in ipairs(NodeData.Connections) do
							local t2 = os.clock()
							local Velocity = ConnectionData.PrevVelocity * ANN.Momentum + LearningRate * (ConnectionData.Gradient / BatchSize)
							ConnectionData.Weight = ConnectionData.Weight - Velocity

							ConnectionData.PrevVelocity = Velocity

							ConnectionData.Gradient = 0

							local Ref = ANN[ConnectionData.ConnectedLayer].Nodes[ConnectionData.ConnectedNode]
							Velocity = Ref.PrevVelocity * ANN.Momentum + LearningRate * (Ref.Gradient / BatchSize)

							Ref.Bias = Ref.Bias - Velocity
							Ref.Gradient = 0
							Ref.PrevVelocity = Velocity
						end
					end
				end

				ANN.NumGradientSteps = 0
			end
		end

		if NumLayers > 2 then
			for i = 1,NumLayers do
				local CurLayer = 'Layer'..tostring(i)

				for j,NodeData in pairs(ANN[CurLayer].Nodes) do
					NodeData.BiasUpdated = false
				end
			end
		else
			for i = NumLayers,NumLayers do
				local CurLayer = 'Layer'..tostring(i)

				for j,NodeData in pairs(ANN[CurLayer].Nodes) do
					NodeData.BiasUpdated = false
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
local Rand = Random.new()
local CopyWeightsEvery = 32
local Gamma = 0.95

--Default parameters
local lr = 0.001--0.0001
local NumInputs = 11
local NumHiddenNodes = 13
local NumHiddenLayers = 3
local NumOutputs = 4
local WeightMin = -0.1
local WeightMax = 0.1
local HiddenLayerActivationFunction = 'TanH'
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

local function CreateNetworks()
	NeuralNetworkFramework.CreateNN('QNetwork',lr,NumInputs,NumHiddenNodes,NumHiddenLayers,NumOutputs,WeightMin,WeightMax,HiddenLayerActivationFunction,OutputLayerActivationFunction,Optimizer,{Momentum=0.99})
	NeuralNetworkFramework.ClipGradients('QNetwork',-1,1)
	NeuralNetworkData.TNetwork = NeuralNetworkData.QNetwork
	Epsilon = 1
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
	
	local X = math.round((Rand:NextNumber(5.5,22.5) - 0.5)) + 0.5
	local Y = 0.5
	local Z = math.round((Rand:NextNumber(-47.5,-29.5) - 0.5) + 0.5)
	
	SnakeHead.CFrame = CFrame.new(X,Y,Z) * CFrame.Angles(0,math.rad(90),0)
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
			
			Reward = -100 --Crashed
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
	--[[local RaycastResults = {}
	local Head = workspace.Snake.SnakeHead
	local RayOrigin = Head.Position
	local RayDirection = (RayOrigin + Head.CFrame.LookVector * 100) - RayOrigin
	local Return = {}
	
	local Params = RaycastParams.new()
	
	Params.FilterDescendantsInstances = {workspace.Apple}
	Params.FilterType = Enum.RaycastFilterType.Blacklist
	
	local Result = workspace:Raycast(RayOrigin,RayDirection,Params)
	local dist = (Result.Position - Head.Position).Magnitude / 24.75883674621582
	
	table.insert(RaycastResults,dist)
	
	RayDirection = (RayOrigin + Head.CFrame.RightVector * -100) - RayOrigin
	
	Result = workspace:Raycast(RayOrigin,RayDirection,Params)
	dist = (Result.Position - Head.Position).Magnitude / 24.75883674621582
	
	table.insert(RaycastResults,dist)
	
	RayDirection = (RayOrigin + Head.CFrame.RightVector * 100) - RayOrigin

	Result = workspace:Raycast(RayOrigin,RayDirection,Params)
	dist = (Result.Position - Head.Position).Magnitude / 24.75883674621582

	table.insert(RaycastResults,dist)
	
	local DistXApple = ((Head.Position.X - workspace.Apple.Position.X) - -17) / 34 -- Min = -17, Max = 17
	local DistZApple = ((Head.Position.Z - workspace.Apple.Position.Z) - -17) / 34
	
	local XVel = 0 -- Right = -X direction
	local YVel = 0 -- Down = -Z/Y direction
	
	if Head.Orientation == Vector3.new(0,90,0) then
		XVel = -1
	elseif Head.Orientation == Vector3.new(0,-90,0) then
		XVel = 1
	elseif Head.Orientation == Vector3.new(0,0,0) then
		YVel = -1
	else
		YVel = 1
	end
	
	Return = {DistXApple,DistZApple,XVel,YVel,RaycastResults[1],RaycastResults[2],RaycastResults[3]}
	
	return Return--]]
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
	
	Params.FilterDescendantsInstances = {Apple}
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
	
	if Head.Orientation == Vector3.new(0,180,0) then
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
	local EpData = {}
	
	while Alive do
		if AmtWait > 0 then
			task.wait(AmtWait)
		end

		Reward = 0

		local Action = math.random(1,4)
		local Inputs = CalculateInputs()

		local Probability = Rand:NextNumber()

		if Probability > Epsilon then
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
				Reward = -1
			else
				Reward = 1
			end
		end

		PrevReward = Reward

		table.insert(EpData,{Cur_s = PrevState,Cur_a = PrevAction,Reward = PrevReward,Next_s = CalculateInputs(),TerminalState = false})
	end

	EpData[#EpData].TerminalState = true
	
	Shuffle(EpData)
	
	table.insert(ExperienceReplayTable,EpData)

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

local function Episode(AmtWait,Debug)
	while Alive do
		if AmtWait > 0 then
			task.wait(AmtWait)
		end

		Reward = 0

		local Action = math.random(1,4)
		local Inputs = CalculateInputs()

		local Probability = Rand:NextNumber()

		if Probability > Epsilon then
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
				Reward = -1
			else
				Reward = 1
			end
		end

		PrevReward = Reward
		
		table.insert(ExperienceReplayTable,{Cur_s = PrevState,Cur_a = PrevAction,Reward = PrevReward,Next_s = CalculateInputs(),TerminalState = false})
	end

	ExperienceReplayTable[#ExperienceReplayTable].TerminalState = true
	
	for k = 1,1 do
		Shuffle(ExperienceReplayTable)

		local Count = 0
		local Clock = os.clock()
		
		for i,v in pairs(ExperienceReplayTable) do
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
				
				if Debug then
					warn('HighestInd: '..HighestIndex)
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
			
			if Debug then
				warn('Target: '..Q_Target)
				warn('QVal: '..Outputs[ActionTaken])	
				warn('Reward: '..Rew)
				warn(Outputs)
			end

			local Error = NeuralNetworkFramework.Backpropagate('QNetwork',Outputs,TargetValues,'MeanSquaredError')
			
			Count += 1

			if Count == CopyWeightsEvery then
				NeuralNetworkData.TNetwork = NeuralNetworkData.QNetwork
			end

			if Clock - os.clock() >= 0.1 then
				task.wait()
				Clock = os.clock()
			end
		end
	end

	for i,v in pairs(workspace.Snake:GetChildren()) do
		v:Destroy()
	end

	StartGame()
	Score = 0
	workspace.Apple:Destroy()
	AddApple()

	Round += 1
	Alive = true

	if Epsilon > 0.01 then
		Epsilon *= Epsilon_Decay
	end

	ExperienceReplayTable = {}
end

local c = os.clock()

local NumDataCollectingEpisodes = 100

for i = 1,NumDataCollectingEpisodes do
	FillExperienceReplayTable(0,false)
	
	if os.clock() - c >= 0.1 then
		task.wait()
		c = os.clock()
		print(math.round(i / NumDataCollectingEpisodes * 100)..'%')
	end
end

print('Going to peform gradient descent in 5 seconds!')

task.wait(5)

print('Start!')

task.wait()

local Count = 0
local Clock = os.clock()
local i = 1

for a,b in pairs(ExperienceReplayTable) do
	local Loss = 0
	local AvgReward = 0
	
	for k,v in pairs(b) do
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
			NeuralNetworkData.TNetwork = NeuralNetworkData.QNetwork
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
end

print('Done!')

ExperienceReplayTable = {}

Epsilon = 0

while true do
	print('Round: '..Round)
	print('Epsilon: '..Epsilon)
	
	Episode(0.1,true)
end
