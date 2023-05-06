import random,copy,pickle,datetime,json,math
from parameters import *
from gameForLearning import *

def rouletteWheel(fitness, bestNum):
    selected_individuals = []
    total_fitness = sum(map(lambda x : (x*1000)**2,fitness))
    
    for _ in range(bestNum):
        random_value = random.random()
        current_sum = 0
        for i in range(len(fitness)):
            current_sum += (fitness[i]*1000)**2 / total_fitness
            if current_sum >= random_value:
                selected_individuals.append(i)
                break
                
    return selected_individuals

innovNum = 0
class SaveAndLoad:
    @staticmethod
    def save(topologies, filename):
        with open(filename+'.neat', 'wb') as _out:
            pickle.dump(topologies, _out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(filename+'.neat', 'rb') as _in:
            return pickle.load(_in)

class Edge:
    def __init__(self,start,end,weight,innov,disAbled = False) -> None:
        self.start = start
        self.end = end
        self.weight = weight
        self.innov = innov
        self.disAbled = disAbled
    
    def __repr__(self) -> str:
        return f"Edge({self.start},{self.end},{self.weight},{self.innov},{self.disAbled})"
        
    def __str__(self) -> str:
        return f"[{self.start},{self.end},{self.weight},{self.innov},{self.disAbled}]"
    
    def __list__(self):
        return [self.start,self.end,self.weight,self.innov,self.disAbled]

class Node:
    def __init__(self,nodeId,before,next,layer) -> None:
        self.nodeId = nodeId
        self.before = before
        self.next = next
        self.layer = layer

def crossOver(t1, t2):
    t3 = Topology(t1.inNodeNum, t1.outNodeNum)
    edges = []
    innov1 = set(map(lambda x : x.innov, t1.edges))
    innov2 = set(map(lambda x : x.innov, t2.edges))
    innovToEdge1 = dict(map(lambda x : (x.innov,copy.copy(x)), t1.edges))
    innovToEdge2 = dict(map(lambda x : (x.innov,copy.copy(x)), t2.edges))
    multi = innov1 & innov2
    
    for i in multi:
        edges.append(random.choice([innovToEdge1[i], innovToEdge2[i]]))
    if t1.fitness > t2.fitness:
        for i in innov1 - innov2:
            edges.append(innovToEdge1[i])
    elif t1.fitness > t2.fitness:
        for i in innov2 - innov1:
            edges.append(innovToEdge2[i])
    else:
        for i in (innov1 - innov2) | (innov2 - innov1):
            if i in innov1:
                edges.append(innovToEdge1[i])
            else:
                edges.append(innovToEdge2[i])
    t3.init(*edges)
    return t3

def adjustFitness(*topologies):
    topologies = copy.copy(topologies)
    sumOfShare = 0
    for i in range(len(topologies)):
        for j in range(len(topologies)):
            if i != j:
                a = topologies[i]
                b = topologies[j]
                a_innov = set(map(lambda x : x.innov, a.edges))
                b_innov = set(map(lambda x : x.innov, b.edges))

                matching_edges = a_innov & b_innov
                disjoint_edges = (a_innov - b_innov) | (b_innov - a_innov)
                N_edges = len(max(a.edges, b.edges, key=len))

                weight_diff = 0
                innovDict1 = dict(map(lambda x: (x.innov, x),a.edges))
                innovDict2 = dict(map(lambda x: (x.innov, x),a.edges))
                for k in matching_edges:
                    weight_diff += abs(innovDict1[k].weight - innovDict2[k].weight)

                delta = C[0] * len(disjoint_edges)/N_edges + C[1] * weight_diff/len(matching_edges)
                sh = int(not delta > DELTA_T)
                sumOfShare += sh
    
    for i in range(len(topologies)):
        topologies[i].fitness /= sumOfShare

    return topologies

class Topology:
    def __init__(self, inNodeNum, outNodeNum) -> None:
        self.inNodeNum = inNodeNum
        self.outNodeNum = outNodeNum
        self.maxLayer = 0
        self.nodes = {}
        self.edges = []
        self.fitness = 0
    
    def init(self,*edges):
        self.maxLayer = 0
        self.nodes = {}
        self.edges = list(edges)
        self.fitness = 0

        def modifyLayer(nextNode):
            for edge in nextNode.next:
                end = edge.end
                if self.nodes[end].layer != -1 and self.nodes[end].layer <= nextNode.layer + 1:
                    self.nodes[end].layer = nextNode.layer + 1
                    nextNode = self.nodes[end]
                    modifyLayer(nextNode)
                else:
                    break
        #print(f'=== edges : {list(map(lambda x: f"{x.start} -> {x.end}",self.edges))}')
        for edge in edges:
            if not edge.start in self.nodes.keys():
                self.nodes[edge.start] = Node(edge.start,[],[edge],0)
            else:
                self.nodes[edge.start].next.append(edge)

            if not edge.end in self.nodes.keys():
                frontLayer = self.nodes[edge.start].layer
                self.nodes[edge.end] = Node(edge.end,[edge],[],frontLayer + 1)
                if self.inNodeNum <= edge.end < self.inNodeNum + self.outNodeNum:
                    self.nodes[edge.end].layer = -1
            else:
                self.nodes[edge.end].before.append(edge)
                frontLayer = self.nodes[edge.start].layer
                if (self.nodes[edge.end].layer < frontLayer + 1 and self.nodes[edge.end].layer != -1):
                    self.nodes[edge.end].layer = frontLayer + 1
                    modifyLayer(self.nodes[edge.end])
        #print(f'=== nodes : {list(map(lambda x: f"{x} : {self.nodes[x].layer}",self.nodes.keys()))}')
        
        for i in range(self.inNodeNum):
            if not i in self.nodes.keys():
                self.nodes[i] = Node(i,[],[],0)
        for i in range(self.inNodeNum,self.outNodeNum + self.inNodeNum):
            if not i in self.nodes.keys():
                self.nodes[i] = Node(i,[],[],-1)
        
    
    def forward(self,*inputValue):
        values = {}
        self.maxLayer = max(map(lambda x: self.nodes[x].layer,self.nodes.keys()))
        for nodeId in self.nodes.keys():
            values[nodeId] = 0
            if nodeId < self.inNodeNum :
                values[nodeId] = inputValue[nodeId]
        def ReLU(x):
            return max(x,0)
        def ELU(x):
            return x if x >= 0 else 0.5*(math.e**x-1)
        for nowLayer in range(self.maxLayer + 1):
            targetLayer = nowLayer + 1
            if nowLayer == self.maxLayer:
                targetLayer = -1
            for edge in self.edges:
                if self.nodes[edge.end].layer == targetLayer and not edge.disAbled:
                    if self.nodes[edge.start].layer != 0:
                        values[edge.end] += ReLU(values[edge.start] * edge.weight)
                    else:
                        values[edge.end] += values[edge.start] * edge.weight
        return [values[i] for i in range(self.inNodeNum , self.inNodeNum + self.outNodeNum)]

    def addEdgeMutation(self):
        global innovNum
        cnt = 0
        for i in list(self.nodes):
            start = self.nodes[i]
            availNodeNum = 0
            connectedZone = list(map(lambda x: x.end ,self.nodes[start.nodeId].next))
            for node in (self.nodes):
                if self.nodes[node].layer > start.layer:
                    availNodeNum += 1
            availNodeNum -= len(connectedZone)
            if availNodeNum <= 0:
                cnt += 1
        if cnt == len(self.nodes)-OUTPUT_NUM:
            self.addNodeMutation()
            return
        availNodeNum = self.outNodeNum
        start = self.nodes[random.choice(list(self.nodes.keys()))]
        connectedZone = list(map(lambda x: x.end ,self.nodes[start.nodeId].next))
        if start.layer != -1 :
            for node in self.nodes.keys():
                if self.nodes[node].layer > start.layer:
                    availNodeNum += 1
        availNodeNum -= len(connectedZone)
        while start.layer == -1 or availNodeNum == 0:
            start = self.nodes[random.choice(list(self.nodes.keys()))]
            availNodeNum = self.outNodeNum
            for node in self.nodes.keys():
                if self.nodes[node].layer > start.layer:
                    availNodeNum += 1
            connectedZone = list(map(lambda x: x.end ,self.nodes[start.nodeId].next))
            availNodeNum -= len(connectedZone)
        
        end = self.nodes[random.choice(list(self.nodes.keys()))]
        cnt = 0
        for i in self.nodes.keys():
            endCheck = self.nodes[i]
            if (endCheck.layer <= start.layer and endCheck.layer != -1) or (endCheck.nodeId in connectedZone):
                cnt += 1
        if cnt == len(self.nodes):
            self.addNodeMutation()
            return
        while (end.layer <= start.layer and end.layer != -1) or (end.nodeId in connectedZone):
            connectedZone = list(map(lambda x: x.end ,self.nodes[start.nodeId].next))
            end = self.nodes[random.choice(list(self.nodes.keys()))]
        weight = random.random()
        #print(f'newEdge({innovNum}) : {start.nodeId} -> {end.nodeId}[{weight}]')
        self.edges.append(Edge(start.nodeId,end.nodeId,weight,innovNum))
        self.init(*self.edges)
        innovNum += 1
    
    def addNodeMutation(self):
        global innovNum

        edgeNum = random.randint(0,len(self.edges)-1)
        edge = self.edges[edgeNum]
        if all(map(lambda x : x.disAbled,self.edges)):
            return
        while edge.disAbled:
            edgeNum = random.randint(0,len(self.edges)-1)
            edge = self.edges[edgeNum]
        self.edges[edgeNum].disAbled = True

        newId = max(self.nodes.keys()) + 1
        frontEdge = Edge(edge.start, newId, 1.0, innovNum)
        endEdge = Edge(newId, edge.end, edge.weight, innovNum)

        self.nodes[edge.start].next.append(frontEdge)
        innovNum += 1

        self.nodes[edge.end].before.append(endEdge)
        innovNum += 1

        self.edges.append(frontEdge)
        self.edges.append(endEdge)
        self.init(*self.edges)
        #print(f'>> newNode : {edge.start} -> ({newId}) -> {edge.end}')
    
    def setWeightMutation(self):
        edgeNum = random.randint(0,len(self.edges)-1)
        edge = self.edges[edgeNum]
        if all(map(lambda x : x.disAbled,self.edges)):
            return
        while edge.disAbled:
            edgeNum = random.randint(0,len(self.edges)-1)
            edge = self.edges[edgeNum]
        newValue = random.random()
        self.edges[edgeNum].weight = newValue
        for i in range(len(self.nodes[edge.start].next)):
            if self.nodes[edge.start].next[i].end == edge.end:
                self.nodes[edge.start].next[i].weight = newValue
                break
        
        for i in range(len(self.nodes[edge.end].next)):
            if self.nodes[edge.end].next[i].start == edge.start:
                self.nodes[edge.end].next[i].weight = newValue
                break
        #print(f">> setWeight({edge.start} -> {edge.end}) : {temp} -> {newValue}")
    
    def addWeightMutation(self):
        #if not all([i.end in tuple(self.nodes) for i in self.edges]):
        #    self.init(*self.edges)
        #    print("????")
        edgeNum = random.randint(0,len(self.edges)-1)
        edge = self.edges[edgeNum]
        if all(map(lambda x : x.disAbled,self.edges)):
            return
        while edge.disAbled:
            edgeNum = random.randint(0,len(self.edges)-1)
            edge = self.edges[edgeNum]
        newValue = (random.random() - 0.5)*2
        temp = self.edges[edgeNum].weight
        self.edges[edgeNum].weight = min(1,max(0,temp + newValue))
        for i in range(len(self.nodes[edge.start].next)):
            if self.nodes[edge.start].next[i].end == edge.end:
                self.nodes[edge.start].next[i].weight = newValue
                break
        
        for i in range(len(self.nodes[edge.end].next)):
            if self.nodes[edge.end].next[i].start == edge.start:
                self.nodes[edge.end].next[i].weight = newValue
                break
        #print(f">> addWeight({edge.start} -> {edge.end}) : {temp} -> {temp + newValue}")

def selection(networks):
    return [networks[i] for i in rouletteWheel(list(map(lambda x : x.fitness,networks)),BEST_NUM)]
    return sorted(networks, key = lambda x : x.fitness, reverse = True)[:BEST_NUM]

def simulation(networks,seed):
    networks = [copy.copy(network) for network in networks]
    fitness = get_fitness(networks,seed)
    print([i/EXPAND for i in fitness])
    #for i in range(len(networks)):print(networks[i].edges, fitness[i])
    for i in range(len(networks)): networks[i].fitness = fitness[i]
    bestOne = max(networks,key = lambda x: x.fitness)
    adjustFitness(*networks)
    return networks, bestOne

def oneCycle(networks,generation = 0,seed = None,printMod = False):
    if printMod : print(f"/////GENERATION {generation+1}/////")
    copyCat = lambda x : [copy.copy(i) for i in x]
    if seed is None: seed = [random.random() for i in range(EXPAND)]
    networks = copyCat(networks)
    networks,bestOne = simulation(networks,seed)
    temp = Topology(OBJECT_NUM,OUTPUT_NUM)
    temp.init(*bestOne.edges)
    temp.fitness = bestOne.fitness
    bestOne = temp
    networks = copyCat(selection(networks))
    new = []
    for i in range(OBJECT_NUM-len(networks)):
        if random.randint(0,3)==0:
            a,b = random.choices(networks,k=2)
            new.append(crossOver(a,b))
        else:
            new.append(random.choice(networks))
            new[-1].init(*new[-1].edges)
        new[-1].fitness = 0
    for i in range(len(new)):
        choice = random.random() * 100
        if choice >= 60:
            new[i].setWeightMutation()
        elif choice >= 20:
            new[i].addWeightMutation()
        elif choice >= 10:
            new[i].addEdgeMutation()
        elif choice >= 0:
            if len(new[i].edges) != 0:
                new[i].addNodeMutation()
            else:
                new[i].addEdgeMutation()
    return copyCat(sorted(networks + new,key = lambda x : x.fitness, reverse= True)),bestOne

def NEAT(generation,file = False,printMod = False,preGene = None):
    global innovNum
    try:
        print("네트워크 만드는중..")
        data = {
                "bestNetwork" : [],
                "bestFitness" : []
            }
        random.seed(SEED)
        seed = [random.random() for i in range(EXPAND)]
        networks = [Topology(INPUT_NUM,OUTPUT_NUM) for i in range(OBJECT_NUM)]
        for i in range(len(networks)):
            edge = []
            preInnovNum = 0
            edge.append(Edge(random.randint(0,INPUT_NUM-1),random.randint(INPUT_NUM,INPUT_NUM+OUTPUT_NUM-1),1,preInnovNum))
            preInnovNum += 1
                #edge.append(Edge(j,INPUT_NUM+1,1,preInnovNum))
                #preInnovNum += 1
            networks[i].init(*edge)
            #innovNum = len(networks) + i * 2
            innovNum += 1
            networks[i].addEdgeMutation()
        if preGene is not None:
            for i,gene in enumerate(preGene):
                networks[i] = Topology(INPUT_NUM, OUTPUT_NUM)
                networks[i].init(*gene)
        #print(list(map(lambda x : x.edges,networks)))
        now = datetime.datetime.now()
        if file:
            print("초기 데이터 설정중..")
            networks,bestOne = simulation(networks,seed)
            data = {"bestNetwork" : [str(bestOne.edges)],
                    "bestFitness" : [bestOne.fitness]
                }
            print("초기데이터 설정 완료!")
        for i in range(generation):
            networks,bestOne = oneCycle(networks, i, seed ,printMod)
            if file:
                data["bestNetwork"].append(str(bestOne.edges))
                data["bestFitness"].append(bestOne.fitness)
            print(f"bestSpecies : {max(networks,key = lambda x: x.fitness).edges}")
            print(f"- Fitness : {max(networks,key = lambda x: x.fitness).fitness}")
            print(f"bestOne : {str(bestOne.edges)}")
            print(f"- Fitness : {bestOne.fitness}")
        if file:
            openFile = open(f'./result/NEAT_{now.year}.{now.month}.{now.day}.{now.hour}.{now.minute}.{now.second}.{now.microsecond}.json','w')
            openFile.write(json.dumps(data,indent=4))
            openFile.close()
    except KeyboardInterrupt:
        if file:
            now = datetime.datetime.now()
            openFile = open(f'./result/NEAT_{now.year}.{now.month}.{now.day}.{now.hour}.{now.minute}.{now.second}.{now.microsecond}.json','w')
            openFile.write(json.dumps(data,indent=4))
            openFile.close()
if __name__ == '__main__':
    preGene = [[Edge(30,43,1,0,True), Edge(9,43,0.13815671679495323,305,False), Edge(44,43,0.31877574280920395,106,False), Edge(42,44,0.947961341782181,322,False)], [Edge(18,43,1,0,True), Edge(44,43,0.33848204957318406,130,True), Edge(45,43,0.8122305610827516,387,True), Edge(26,46,0.03829580383680464,521,False), Edge(33,46,-0.04194512738158318,663,False), Edge(46,43,1,411,True), Edge(48,46,0.36649879892180737,539,False), Edge(46,43,-0.6868148548112025,424,False), Edge(28,46,0.2602031424128599,435,True), Edge(49,43,0.798773427008344,571,False), Edge(1,43,-0.8310601675715232,322,False), Edge(17,43,0.18522165391615453,326,True), Edge(2,46,0.41279723155677805,468,True), Edge(0,43,0.946119828466139,350,False), Edge(29,46,0.7535841431108655,479,True), Edge(7,46,0.1279826832322194,611,True), Edge(47,43,0.798773427008344,495,True), Edge(50,46,-0.7668817043847558,626,False), Edge(45,43,0.2643617656468146,373,True)]]
    NEAT(1000,True,printMod=True,preGene=preGene)