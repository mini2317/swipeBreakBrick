import random,copy,pickle,datetime,json
from parameters import *
from gameForLearning import *
import numpy as np

def rouletteWheel(fitness, bestNum):
    selected_individuals = []
    total_fitness = sum(fitness)
    
    for _ in range(bestNum):
        random_value = random.random()
        current_sum = 0
        for i in range(len(fitness)):
            current_sum += fitness[i] / total_fitness
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
        newValue = (random.random() - 0.5)/2
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
    a = sorted(networks, key = lambda x : x.fitness, reverse = True)
    return [a[i] for i in rouletteWheel(list(map(lambda x : x.fitness,a)),BEST_NUM)]
    return sorted(networks, key = lambda x : x.fitness, reverse = True)[:BEST_NUM]

def simulation(networks,seed):
    networks = [copy.copy(network) for network in networks]
    fitness = get_fitness(networks,seed)
    print(fitness)
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
        if choice >= 70:
            new[i].setWeightMutation()
        elif choice >= 30:
            new[i].addWeightMutation()
        elif choice >= 20:
            new[i].addEdgeMutation()
        elif choice >= 10:
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
            networks[0] = Topology(INPUT_NUM, OUTPUT_NUM)
            networks[0].init(*preGene)
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
                data["bestNetwork"].append(bestOne.edges)
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
    preGene = [Edge(42,45,-0.20850536903032707,0,True), Edge(49,48,0.6991917655239416,384,False), Edge(46,45,0.8394768382835704,130,True), Edge(50,49,1.0,386,True), Edge(2,45,-0.0031078858943217047,394,False), Edge(29,46,-0.17760535469493094,395,True), Edge(46,45,1,268,True), Edge(34,46,0.7518213174737185,404,False), Edge(20,46,-0.1527284790449311,405,False), Edge(46,45,0.8394768382835704,289,False), Edge(47,45,0.6802212558900103,418,False), Edge(8,45,0.8743950412148368,420,False), Edge(46,45,1,166,True), Edge(1,45,0.8532457941804819,424,True), Edge(39,45,-0.22230579887289553,553,False), Edge(51,49,0.1902892869210701,426,False), Edge(46,45,0.7181287826235541,172,True), Edge(11,46,0.5577741227646452,438,False), Edge(47,45,1,314,True), Edge(11,45,0.5561312622788788,576,False), Edge(49,46,-0.2292413854251934,453,True), Edge(48,45,0.6606885204455033,334,False), Edge(30,45,-0.2201204593796422,463,False), Edge(42,45,0.9445233495537,466,False), Edge(36,45,0.23207596769709676,339,True), Edge(33,48,0.23633072070245698,594,False), Edge(27,46,0.023589312803096507,471,True), Edge(53,46,-0.07363442345927707,599,True), Edge(47,45,0.04530491626484062,346,True), Edge(48,45,0.22202818303422972,353,True), Edge(47,45,0.6062009872792029,356,True), Edge(46,45,-0.1369720697311655,109,True), Edge(43,46,0.7509950613509226,372,True), Edge(2,45,0.5474964354003338,373,False), Edge(52,45,-0.133619105815325,509,False), Edge(17,48,0.14458734445373633,638,False), Edge(54,45,0.8743950412148368,783,False), Edge(19,46,0.2811067580489678,786,True), Edge(12,48,0.16498507950753494,658,False), Edge(54,45,-0.23265827230045089,671,False), Edge(6,49,0.29657119663306764,805,False), Edge(56,46,0.2811067580489678,809,False), Edge(2,49,0.27030454018235195,685,False), Edge(31,48,0.1221529766242529,557,False), Edge(54,45,-0.15410220836403843,687,False), Edge(3,48,0.5081100077472492,944,False), Edge(27,45,0.8237806920171712,817,False), Edge(58,48,0.2195185179489555,953,False), Edge(55,45,0.1929170815011203,705,False), Edge(58,46,0.37436712555137375,961,False), Edge(3,49,0.0742099663878123,969,False), Edge(21,49,0.023675986746312605,588,False), Edge(57,46,-0.003506582401114733,850,False), Edge(35,54,0.2387459433337682,742,False), Edge(36,49,-0.23360614923598244,750,False)]
    NEAT(100,True,printMod=True,preGene=preGene)