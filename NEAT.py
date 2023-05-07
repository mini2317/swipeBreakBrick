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
                    values[edge.end] += values[edge.start] * edge.weight
                    #if self.nodes[edge.start].layer != 0:
                    #     values[edge.end] += ReLU(values[edge.start] * edge.weight)
                    # else:
                    #    values[edge.end] += values[edge.start] * edge.weight
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
        newValue = 2*(random.random()-0.5)
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
        self.edges[edgeNum].weight = min(1,max(-1,temp + newValue))
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
            try:
                a,b = random.choices(networks,k=2)
                new.append(crossOver(a,b))
            except:
                new.append(random.choice(networks))
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
            #random.seed(seed[0])
            #seed = [random.random() for i in range(EXPAND)]
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
    preGene = [
        [Edge(16,36,1,0,True), Edge(50,48,0.9679691697549194,2049,False), Edge(91,37,0.12967116897126774,2051,False), Edge(91,36,0.47822390502148004,2074,False), Edge(92,48,0.9792693085156601,2076,False), Edge(37,36,0.12307851440160733,46,True), Edge(93,78,0.5865861224878542,2132,True), Edge(37,36,1,97,True), Edge(92,39,0.24952337756836362,2152,False), Edge(93,37,0.7457428693554302,2163,False), Edge(91,87,0.9357686206976068,2171,False), Edge(93,48,0.42866970280661376,2177,False), Edge(26,48,0.10894550119257107,2200,False), Edge(94,87,0.23295866897799922,2201,True), Edge(37,36,0.2555094646175703,157,True), Edge(34,54,-0.07010372955148392,2205,False), Edge(93,48,-0.4523442394355155,2214,False), Edge(94,54,0.3992677087462253,2223,False), Edge(69,39,0.7432813763712653,2234,False), Edge(27,49,-0.1232948311572819,2240,True), Edge(94,54,0.3992677087462253,2245,False), Edge(69,48,0.10750034172974532,2257,False), Edge(71,49,0.4337521845774851,2263,False), Edge(95,78,-0.39972968893878313,2264,False), Edge(89,48,0.12249921624037752,2286,False), Edge(37,36,1,244,True), Edge(94,78,-0.9044972991167137,2300,False), Edge(37,36,0.42676425202570956,259,False), Edge(63,54,0.6431584439039271,2307,False), Edge(37,36,0.9272693056268737,268,False), Edge(65,37,0.6929162324590197,2317,False), Edge(95,48,0.2905311220101552,2318,False), Edge(95,49,-0.5109531612261211,2342,False), Edge(52,87,0.19795411872734736,2344,False), Edge(37,36,0.6965624474859198,298,True), Edge(38,36,-0.7249711491713702,302,False), Edge(96,39,-0.47479626650104567,2351,False), Edge(16,39,0.7597861335895142,320,False), Edge(63,36,-0.1310605615076461,322,False), Edge(61,81,-0.36169608461516023,2370,False), Edge(97,49,0.37407567433336975,2375,False), Edge(22,36,-0.9255772972135201,333,False), Edge(63,36,0.8192624571415981,336,False), Edge(38,37,0.5616260727066895,338,True), Edge(23,54,0.7845467222343699,346,True), Edge(24,48,0.2610689625089182,349,False), Edge(39,36,-0.4496946419832797,353,True), Edge(64,36,0.028812382922944524,355,False), Edge(96,36,-0.9181837764361287,2404,False), Edge(65,36,0.03717027728480371,359,False), Edge(6,36,0.17397474248127132,374,True), Edge(40,39,0.7486775375863762,385,True), Edge(96,36,0.03094818394171517,2434,False), Edge(41,36,0.8196341648297829,387,True), Edge(64,54,0.7845467222343699,391,True), Edge(47,36,0.9272693056268737,395,True), Edge(40,36,-0.021039102074985605,398,True), Edge(66,36,-0.1411865355904658,401,False), Edge(40,36,-0.22727225089644887,403,True), Edge(12,54,-0.6254203702321586,407,True), Edge(41,36,0.6360675157869415,416,False), Edge(98,78,0.3929434251033561,2466,False), Edge(42,39,-0.29542648698059004,2468,False), Edge(29,39,-0.9479014055095569,424,True), Edge(27,37,0.8088518430566118,428,True), Edge(40,36,-0.021039102074985605,430,True), Edge(48,36,0.6903649795829547,439,False), Edge(6,37,0.9967530259749053,440,False), Edge(10,39,0.7732261879326431,441,True), Edge(48,36,-0.40669043820716033,443,False), Edge(33,37,-0.08287174811297371,447,True), Edge(97,78,-0.5804240053638561,2498,False), Edge(66,37,0.5029298588831674,451,False), Edge(79,87,0.43102974637419,2500,False), Edge(97,39,0.28088403047692934,2504,False), Edge(99,77,0.19519840928031162,2506,False), Edge(42,36,-0.7014444413534948,459,True), Edge(19,54,0.07432979317369237,462,True), Edge(49,36,-0.4238627588717503,463,True), Edge(67,37,-0.123591026422887,464,False), Edge(98,54,0.7221856675217535,2512,True), Edge(23,48,0.09755292303815444,2515,False), Edge(48,36,-0.9432525189286147,479,False), Edge(0,37,0.8692034596775957,483,False), Edge(50,37,0.2976123467309044,487,True), Edge(73,77,0.7135266761826092,2536,False), Edge(100,36,-0.5821594446889009,2539,False), Edge(6,37,0.2899804310606785,492,True), Edge(59,49,-0.39748565960729776,2544,False), Edge(98,36,0.03094818394171517,2545,False), Edge(43,36,-0.8459905063536599,499,True), Edge(101,54,0.20444200602000495,2555,True), Edge(41,36,0.2154248654415527,509,False), Edge(6,39,0.8386498259372221,520,False), Edge(49,36,-0.6158664189451919,525,True), Edge(102,37,-0.43503737666300046,2574,False), Edge(28,36,0.6737382036759751,531,False), Edge(50,37,0.5757705304464205,532,True), Edge(19,37,0.6204277756795049,535,False), Edge(34,37,0.5455689424840654,2587,False), Edge(40,48,0.3841901115445554,543,False), Edge(44,36,-0.318871195041148,546,True), Edge(49,39,0.15474225915109896,547,True), Edge(51,39,0.3044822132751166,551,False), Edge(46,37,0.18647850057134585,553,True), Edge(38,77,0.7598903060247231,2603,False), Edge(99,54,0.7372792819456198,2605,False), Edge(73,99,0.9571134675934222,2607,False), Edge(32,39,0.6229121163827249,577,False), Edge(43,37,0.49205810720018306,578,True), Edge(51,37,0.8088518430566118,579,True), Edge(67,54,0.1951007245550983,2629,True), Edge(68,36,-0.021039102074985605,586,True), Edge(3,39,0.9514983128252149,589,True), Edge(100,36,-0.6158664189451919,2639,False), Edge(40,49,0.3673087034485545,593,False), Edge(30,36,-0.21084161144019475,594,False), Edge(48,36,-0.5257909877238525,595,True), Edge(47,48,0.6584321854037208,603,True), Edge(68,39,-0.16843477841414645,604,False), Edge(101,49,0.5638759426360338,2666,False), Edge(103,49,0.9709622470027257,2676,False), Edge(52,37,0.6217805232551452,633,True), Edge(64,37,-0.7018093072249494,636,False), Edge(27,54,-0.06472412668119976,643,False), Edge(69,36,0.6769775276258676,644,True), Edge(51,48,0.9417534477992975,646,False), Edge(41,54,0.5980293099361317,647,True), Edge(69,49,-0.7179995372782297,653,False), Edge(48,36,0.9559533885581981,668,False), Edge(102,36,-0.7638533973570822,2724,False), Edge(45,36,-0.07895095253087048,684,True), Edge(104,78,0.5865861224878542,2749,False), Edge(69,54,0.3096801068893904,705,False), Edge(53,37,0.12341540956682451,708,False), Edge(103,39,0.5904805432857121,2760,False), Edge(27,36,-0.4947984123112614,718,False), Edge(49,37,-0.20213795065088713,724,True), Edge(69,37,0.06334118461519522,729,False), Edge(48,39,0.9735427239600714,733,True), Edge(104,36,0.6629961922888061,2783,False), Edge(42,48,-0.5357384674653098,740,False), Edge(50,36,-0.9189911588578752,741,False), Edge(15,48,0.0609814731996825,746,True), Edge(37,39,0.8510031486544098,2796,False), Edge(105,36,0.8770136442518929,2797,False), Edge(69,54,0.6512900529302501,750,True), Edge(106,36,0.8630079946018181,2801,False), Edge(46,36,0.9425030441229787,757,True), Edge(70,39,-0.7241689671198921,758,False), Edge(87,37,0.48270368382170215,2812,False), Edge(70,37,0.9352008356887924,776,False), Edge(50,36,-0.31780587598548293,778,False), Edge(51,49,-0.861380995547135,791,False), Edge(105,37,-0.16156693597102834,2850,False), Edge(80,77,0.5518855436125463,2863,False), Edge(54,36,0.12434244685395823,820,True), Edge(51,36,-0.5615210108751827,825,True), Edge(71,37,-0.3166746999466068,831,False), Edge(88,99,0.6883351794749062,2880,False), Edge(31,37,-0.12610978329693223,835,False), Edge(46,39,0.6881431340485101,844,True), Edge(72,36,0.4040487403333768,856,False), Edge(68,54,-0.496908389346427,863,False), Edge(86,87,-0.35440934023376824,2916,False), Edge(107,36,0.5689533297934748,2917,True), Edge(21,54,-0.7042177540003556,872,False), Edge(20,36,0.32235576244518316,875,False), Edge(71,36,-0.9405826036485603,892,False), Edge(71,36,0.6799222938352465,896,True), Edge(73,36,0.9703340759346426,898,False), Edge(18,37,0.08671669617684874,900,True), Edge(70,36,0.14386466895588956,906,False), Edge(55,36,0.1028222288030296,907,True), Edge(42,37,0.24933978575832327,911,False), Edge(64,49,0.6851991968516669,922,False), Edge(62,54,0.8941170676594443,923,False), Edge(54,39,-0.18923272502332633,926,True), Edge(52,49,0.5782329042878998,931,False), Edge(49,77,0.03369200703881248,2984,False), Edge(70,49,0.9414942957082036,937,True), Edge(72,49,0.7924538330731987,940,False), Edge(28,39,0.589512129302477,943,False), Edge(72,36,0.8930783645544096,944,True), Edge(106,54,0.9824176035915613,2992,False), Edge(47,49,0.1899145862261612,947,True), Edge(21,37,-0.16156693597102834,948,False), Edge(34,49,-0.5809061845330319,955,True), Edge(73,36,0.9703340759346426,959,False), Edge(107,54,0.5267100171182313,3008,False), Edge(55,36,0.5258863264859874,961,False), Edge(72,36,1,964,False), Edge(66,49,-0.1354985508254598,3024,False), Edge(72,37,0.2976123467309044,978,True), Edge(40,99,0.3013978719526301,3028,False), Edge(108,39,0.04779982813038397,3029,False), Edge(28,48,-0.853193070802835,983,False), Edge(74,39,0.005726280127722916,989,False), Edge(106,36,1,3039,False), Edge(74,39,0.0515825932676095,1001,True), Edge(109,54,0.02095269324182525,3058,False), Edge(7,39,0.3161055749535986,1013,False), Edge(44,78,-0.6449803820983862,3061,False), Edge(34,48,0.7197811690610363,1015,True), Edge(1,78,0.6258734448444967,3066,False), Edge(108,36,0.596270410300471,3067,False), Edge(75,37,0.5294891371403418,1023,True), Edge(109,37,0.8517384667636347,3072,False), Edge(74,54,-0.9883092020061461,1025,False), Edge(74,36,-0.20351815441442156,1029,False), Edge(76,99,0.15137250773277022,3079,False), Edge(108,37,0.12341540956682451,3080,False), Edge(23,36,0.8728329585427566,1035,False), Edge(71,37,-0.9303942608921612,1036,True), Edge(106,77,-0.010044462550975464,3088,False), Edge(56,87,0.4770027831024106,3092,False), Edge(107,36,-0.21084161144019475,3093,False), Edge(90,77,0.14041706805172716,3096,False), Edge(53,54,-0.48833909782057505,1055,False), Edge(109,77,-0.8512444232608283,3108,False), Edge(56,36,0.9880295265779211,1062,False), Edge(106,36,0.5402867161899334,3110,False), Edge(75,36,-0.2466762726062015,1064,False), Edge(76,36,0.5616274901734499,1067,False), Edge(57,36,-0.6440467910393914,1071,True), Edge(43,48,-0.948844064224063,1073,False), Edge(110,49,-0.6523281099821265,3125,False), Edge(56,48,-0.4749878742598257,1078,True), Edge(88,39,-0.9779891802416121,3128,False), Edge(76,48,-0.7922756988590154,1081,True), Edge(71,54,0.6151253175494457,1083,True), Edge(58,36,-0.888586056035876,1085,False), Edge(111,48,0.4387163410835342,3144,False), Edge(77,39,0.1999769856424527,1103,False), Edge(78,39,-0.6959478885088008,1116,False), Edge(76,49,0.9709622470027257,1118,True), Edge(58,37,0.10778315608448508,1120,False), Edge(112,39,0.15474225915109896,3168,False), Edge(59,36,-0.6875440737399712,1122,True), Edge(75,36,-0.8545184101908228,1124,False), Edge(40,54,0.4547184589695824,3176,False), Edge(58,36,-0.5232040357664516,1129,False), Edge(59,36,-0.2347818349668831,1131,False), Edge(110,36,0.5689533297934748,3186,False), Edge(28,49,0.7433195201510598,3188,False), Edge(80,37,0.12646518139423457,1143,False), Edge(43,54,0.02095269324182525,1146,False), Edge(77,36,-0.13608537042692426,1156,True), Edge(59,48,0.98682736526802,1157,True), Edge(33,36,-0.07972507782170624,1160,False), Edge(26,36,-0.6864186564566612,1166,False), Edge(76,77,0.7133368701083049,1178,False), Edge(32,37,-0.5516753124265643,1190,False), Edge(60,36,-0.0036955812036809643,1191,False), Edge(110,54,-0.8556207055581007,3240,False), Edge(99,36,0.4522786332566828,3244,False), Edge(109,39,0.31121082366373054,3268,False), Edge(76,54,0.19180957155630085,1223,False), Edge(34,39,0.11493609280602635,1227,True), Edge(111,36,-0.2466762726062015,3275,False), Edge(79,78,-0.39972968893878313,1234,True), Edge(77,37,-0.3761706947316179,1245,False), Edge(44,48,0.6242975464990579,1247,False), Edge(112,36,-0.3735695828627672,3298,False), Edge(61,37,-0.8918878207459542,1251,False), Edge(42,78,0.15624276814445826,1254,True), Edge(113,78,-0.48205822377746,3307,False), Edge(113,36,-0.13608537042692426,3319,False), Edge(57,49,0.8417957701447749,1285,True), Edge(45,49,-0.6523281099821265,1296,False), Edge(81,39,0.28088403047692934,1307,False), Edge(60,39,-0.16843477841414645,1308,True), Edge(114,39,0.589512129302477,3361,False), Edge(61,48,0.8603991723606321,1315,True), Edge(114,87,0.23295866897799922,3363,False), Edge(26,36,0.8770136442518929,1318,False), Edge(115,54,0.1951007245550983,3367,False), Edge(62,81,-0.09699215384744653,1325,False), Edge(65,99,0.9811165929509954,3381,False), Edge(115,54,0.7372792819456198,3387,False), Edge(61,36,0.5974955150106886,3401,False), Edge(116,36,0.9844383962496319,3439,False), Edge(81,37,-0.6413515373899386,1395,False), Edge(82,37,0.5268319498165015,1421,False), Edge(62,37,-0.3570162942374415,1423,False), Edge(75,77,-0.5307110096314154,1432,False), Edge(117,87,0.3970083915245821,3488,False), Edge(100,39,0.28224346991931526,3565,True), Edge(22,77,0.9422907813758393,3568,False), Edge(83,49,0.3311963506327551,1527,False), Edge(83,36,0.7465021704738459,1543,False), Edge(16,87,0.24224367350706866,3592,False), Edge(84,37,0.5057481299313846,1549,False), Edge(84,36,-0.5692669168656832,1559,False), Edge(98,37,0.9870987102000552,3636,True), Edge(83,48,0.16296763735451303,1592,False), Edge(52,39,0.3355342754013755,1594,False), Edge(81,36,0.7042535162522163,1627,False), Edge(85,39,-0.38323813134329,1649,False), Edge(85,81,-0.25511575372551953,1653,False), Edge(85,36,-0.3735695828627672,1668,False), Edge(85,48,0.812894073355199,1691,False), Edge(86,37,-0.7593877351301184,1697,False), Edge(68,49,0.2942468328234755,1720,False), Edge(86,36,0.05925037023826052,1721,False), Edge(87,36,0.9917329972373357,1740,True), Edge(87,78,-0.39972968893878313,1744,False), Edge(27,87,0.5783664228546296,1793,False), Edge(40,78,0.13287445951599386,1798,False), Edge(65,81,0.7255972240388604,1813,False), Edge(88,37,-0.5032477120999719,1834,False), Edge(88,36,0.03094818394171517,1836,True), Edge(89,78,-0.11848825291342213,1846,True), Edge(89,54,0.952673298008416,1852,True), Edge(87,39,-0.8186732830398622,1855,False), Edge(60,54,0.5213724026243489,1897,False), Edge(79,48,-0.9945280974270221,1899,False), Edge(90,39,0.9425284113836925,1905,False), Edge(43,49,0.9773998934653928,1923,False), Edge(91,36,0.6769775276258676,1928,True), Edge(34,81,-0.3635390032185555,1933,False), Edge(90,49,0.44760706183174626,1950,False), Edge(90,36,0.14423856701348314,1962,False), Edge(92,49,0.2364405843527535,1983,False), Edge(24,78,0.6580511038327919,1992,False), Edge(90,37,0.5294891371403418,2002,False), Edge(58,49,-0.5975971020552444,2041,False), Edge(71,99,0.11873118475853706,3672,False), Edge(19,118,1.0,3695,False), Edge(118,54,0.07432979317369237,3695,False), Edge(48,119,1.0,3709,False), Edge(119,39,0.9735427239600714,3709,False)],
        [Edge(16,36,1,0,True), Edge(50,48,0.9679691697549194,2049,False), Edge(91,37,0.12967116897126774,2051,False), Edge(91,36,0.47822390502148004,2074,False), Edge(92,48,0.9792693085156601,2076,False), Edge(37,36,0.12307851440160733,46,True), Edge(93,78,0.5865861224878542,2132,True), Edge(37,36,1,97,True), Edge(92,39,0.24952337756836362,2152,False), Edge(93,37,0.7457428693554302,2163,False), Edge(91,87,0.9357686206976068,2171,False), Edge(93,48,0.42866970280661376,2177,False), Edge(26,48,0.10894550119257107,2200,False), Edge(94,87,0.23295866897799922,2201,True), Edge(37,36,0.2555094646175703,157,True), Edge(34,54,-0.07010372955148392,2205,False), Edge(93,48,-0.4523442394355155,2214,False), Edge(94,54,0.3992677087462253,2223,False), Edge(69,39,0.7432813763712653,2234,False), Edge(27,49,-0.1232948311572819,2240,True), Edge(94,54,0.3992677087462253,2245,False), Edge(69,48,0.10750034172974532,2257,False), Edge(71,49,0.4337521845774851,2263,False), Edge(95,78,-0.39972968893878313,2264,False), Edge(89,48,0.12249921624037752,2286,False), Edge(37,36,1,244,True), Edge(94,78,-0.9044972991167137,2300,False), Edge(37,36,0.42676425202570956,259,False), Edge(63,54,0.6431584439039271,2307,False), Edge(37,36,0.9272693056268737,268,False), Edge(65,37,0.6929162324590197,2317,False), Edge(95,48,0.2905311220101552,2318,False), Edge(95,49,-0.5109531612261211,2342,False), Edge(52,87,0.19795411872734736,2344,False), Edge(37,36,0.6965624474859198,298,True), Edge(38,36,-0.7249711491713702,302,False), Edge(96,39,-0.47479626650104567,2351,False), Edge(16,39,0.7597861335895142,320,False), Edge(63,36,-0.1310605615076461,322,False), Edge(61,81,-0.36169608461516023,2370,False), Edge(97,49,0.37407567433336975,2375,False), Edge(22,36,-0.9255772972135201,333,False), Edge(63,36,0.8192624571415981,336,False), Edge(38,37,0.5616260727066895,338,True), Edge(23,54,0.7845467222343699,346,True), Edge(24,48,0.2610689625089182,349,False), Edge(39,36,-0.4496946419832797,353,True), Edge(64,36,0.028812382922944524,355,False), Edge(96,36,-0.9181837764361287,2404,False), Edge(65,36,0.03717027728480371,359,False), Edge(6,36,0.17397474248127132,374,True), Edge(40,39,0.7486775375863762,385,True), Edge(96,36,0.03094818394171517,2434,False), Edge(41,36,0.8196341648297829,387,True), Edge(64,54,0.7845467222343699,391,True), Edge(47,36,0.9272693056268737,395,True), Edge(40,36,-0.021039102074985605,398,True), Edge(66,36,-0.1411865355904658,401,False), Edge(40,36,-0.22727225089644887,403,True), Edge(12,54,-0.6254203702321586,407,True), Edge(41,36,0.6360675157869415,416,False), Edge(98,78,0.3929434251033561,2466,False), Edge(42,39,-0.29542648698059004,2468,False), Edge(29,39,-0.9479014055095569,424,True), Edge(27,37,0.8088518430566118,428,True), Edge(40,36,-0.021039102074985605,430,True), Edge(48,36,0.6903649795829547,439,False), Edge(6,37,0.9967530259749053,440,False), Edge(10,39,0.7732261879326431,441,True), Edge(48,36,-0.40669043820716033,443,False), Edge(33,37,-0.08287174811297371,447,True), Edge(97,78,-0.5804240053638561,2498,False), Edge(66,37,0.5029298588831674,451,False), Edge(79,87,0.43102974637419,2500,False), Edge(97,39,0.28088403047692934,2504,False), Edge(99,77,0.19519840928031162,2506,False), Edge(42,36,-0.7014444413534948,459,True), Edge(19,54,0.07432979317369237,462,True), Edge(49,36,-0.4238627588717503,463,True), Edge(67,37,-0.123591026422887,464,False), Edge(98,54,0.7221856675217535,2512,True), Edge(23,48,0.09755292303815444,2515,False), Edge(48,36,-0.9432525189286147,479,False), Edge(0,37,0.8692034596775957,483,False), Edge(50,37,0.2976123467309044,487,True), Edge(73,77,0.7135266761826092,2536,False), Edge(100,36,-0.5821594446889009,2539,False), Edge(6,37,0.2899804310606785,492,True), Edge(59,49,-0.39748565960729776,2544,False), Edge(98,36,0.03094818394171517,2545,False), Edge(43,36,-0.8459905063536599,499,True), Edge(101,54,0.20444200602000495,2555,False), Edge(41,36,0.2154248654415527,509,False), Edge(6,39,0.8386498259372221,520,False), Edge(49,36,-0.6158664189451919,525,True), Edge(102,37,-0.43503737666300046,2574,False), Edge(28,36,0.6737382036759751,531,False), Edge(50,37,0.5757705304464205,532,True), Edge(19,37,0.6204277756795049,535,False), Edge(34,37,0.5455689424840654,2587,False), Edge(40,48,0.3841901115445554,543,False), Edge(44,36,-0.318871195041148,546,True), Edge(49,39,0.15474225915109896,547,True), Edge(51,39,0.3044822132751166,551,False), Edge(46,37,0.18647850057134585,553,True), Edge(38,77,0.7598903060247231,2603,False), Edge(99,54,0.7372792819456198,2605,False), Edge(73,99,0.9571134675934222,2607,False), Edge(32,39,0.6229121163827249,577,False), Edge(43,37,0.49205810720018306,578,True), Edge(51,37,0.8088518430566118,579,True), Edge(67,54,0.1951007245550983,2629,True), Edge(68,36,-0.021039102074985605,586,True), Edge(3,39,0.9514983128252149,589,True), Edge(100,36,-0.6158664189451919,2639,False), Edge(40,49,0.3673087034485545,593,False), Edge(30,36,-0.21084161144019475,594,False), Edge(48,36,-0.5257909877238525,595,True), Edge(47,48,0.6584321854037208,603,True), Edge(68,39,-0.16843477841414645,604,False), Edge(101,49,0.5638759426360338,2666,False), Edge(103,49,0.9709622470027257,2676,False), Edge(52,37,0.6217805232551452,633,True), Edge(64,37,-0.7018093072249494,636,False), Edge(27,54,-0.06472412668119976,643,False), Edge(69,36,0.6769775276258676,644,True), Edge(51,48,0.9417534477992975,646,False), Edge(41,54,0.5980293099361317,647,True), Edge(69,49,-0.7179995372782297,653,False), Edge(48,36,0.9559533885581981,668,False), Edge(102,36,-0.7638533973570822,2724,False), Edge(45,36,-0.07895095253087048,684,True), Edge(104,78,0.5865861224878542,2749,False), Edge(69,54,0.3096801068893904,705,False), Edge(53,37,0.12341540956682451,708,False), Edge(103,39,0.5904805432857121,2760,False), Edge(27,36,-0.4947984123112614,718,False), Edge(49,37,-0.20213795065088713,724,True), Edge(69,37,0.06334118461519522,729,False), Edge(48,39,0.9735427239600714,733,True), Edge(104,36,0.6629961922888061,2783,False), Edge(42,48,-0.5357384674653098,740,False), Edge(50,36,-0.9189911588578752,741,False), Edge(15,48,0.0609814731996825,746,True), Edge(37,39,0.8510031486544098,2796,False), Edge(105,36,0.8770136442518929,2797,False), Edge(69,54,0.6512900529302501,750,True), Edge(106,36,-0.912799608059162,2801,False), Edge(46,36,0.9425030441229787,757,True), Edge(70,39,-0.7241689671198921,758,False), Edge(87,37,0.48270368382170215,2812,False), Edge(70,37,0.9352008356887924,776,False), Edge(50,36,-0.31780587598548293,778,False), Edge(51,49,-0.861380995547135,791,False), Edge(105,37,-0.16156693597102834,2850,False), Edge(80,77,0.5518855436125463,2863,False), Edge(54,36,0.12434244685395823,820,True), Edge(51,36,-0.5615210108751827,825,True), Edge(71,37,-0.3166746999466068,831,False), Edge(88,99,0.6883351794749062,2880,False), Edge(31,37,-0.12610978329693223,835,False), Edge(46,39,0.6881431340485101,844,True), Edge(72,36,0.4040487403333768,856,False), Edge(68,54,-0.496908389346427,863,False), Edge(86,87,-0.35440934023376824,2916,False), Edge(107,36,0.5689533297934748,2917,True), Edge(21,54,-0.7042177540003556,872,False), Edge(20,36,0.32235576244518316,875,False), Edge(71,36,-0.9405826036485603,892,False), Edge(71,36,0.6799222938352465,896,True), Edge(73,36,0.9703340759346426,898,False), Edge(18,37,0.08671669617684874,900,True), Edge(70,36,0.14386466895588956,906,False), Edge(55,36,0.1028222288030296,907,True), Edge(42,37,0.24933978575832327,911,False), Edge(64,49,0.6851991968516669,922,False), Edge(62,54,0.8941170676594443,923,False), Edge(54,39,-0.18923272502332633,926,True), Edge(52,49,0.5782329042878998,931,False), Edge(49,77,0.03369200703881248,2984,False), Edge(70,49,0.9414942957082036,937,True), Edge(72,49,0.7924538330731987,940,False), Edge(28,39,0.589512129302477,943,False), Edge(72,36,0.8930783645544096,944,True), Edge(106,54,0.9824176035915613,2992,False), Edge(47,49,0.1899145862261612,947,True), Edge(21,37,-0.16156693597102834,948,False), Edge(34,49,-0.5809061845330319,955,True), Edge(73,36,0.9703340759346426,959,False), Edge(107,54,0.5267100171182313,3008,False), Edge(55,36,0.5258863264859874,961,False), Edge(72,36,1,964,False), Edge(66,49,-0.1354985508254598,3024,False), Edge(72,37,0.2976123467309044,978,True), Edge(40,99,0.3013978719526301,3028,False), Edge(108,39,0.04779982813038397,3029,False), Edge(28,48,-0.853193070802835,983,False), Edge(74,39,0.005726280127722916,989,False), Edge(106,36,0.6829265888515275,3039,False), Edge(74,39,0.0515825932676095,1001,True), Edge(109,54,0.02095269324182525,3058,False), Edge(7,39,0.3161055749535986,1013,False), Edge(44,78,-0.6449803820983862,3061,False), Edge(34,48,0.7197811690610363,1015,True), Edge(1,78,0.6258734448444967,3066,False), Edge(108,36,0.596270410300471,3067,False), Edge(75,37,0.5294891371403418,1023,True), Edge(109,37,0.8517384667636347,3072,False), Edge(74,54,-0.9883092020061461,1025,False), Edge(74,36,-0.20351815441442156,1029,False), Edge(76,99,0.15137250773277022,3079,False), Edge(108,37,0.12341540956682451,3080,False), Edge(23,36,0.8728329585427566,1035,False), Edge(71,37,-0.9303942608921612,1036,True), Edge(106,77,-0.010044462550975464,3088,False), Edge(56,87,0.4770027831024106,3092,False), Edge(107,36,-0.21084161144019475,3093,False), Edge(90,77,0.14041706805172716,3096,False), Edge(53,54,-0.48833909782057505,1055,False), Edge(109,77,-0.8512444232608283,3108,False), Edge(56,36,0.9880295265779211,1062,False), Edge(106,36,0.5402867161899334,3110,False), Edge(75,36,-0.2466762726062015,1064,False), Edge(76,36,0.5616274901734499,1067,False), Edge(57,36,-0.6440467910393914,1071,True), Edge(43,48,-0.948844064224063,1073,False), Edge(110,49,-0.6523281099821265,3125,False), Edge(56,48,-0.4749878742598257,1078,True), Edge(88,39,-0.9779891802416121,3128,False), Edge(76,48,-0.7922756988590154,1081,True), Edge(71,54,0.6151253175494457,1083,True), Edge(58,36,-0.888586056035876,1085,False), Edge(111,48,0.4387163410835342,3144,False), Edge(77,39,0.1999769856424527,1103,False), Edge(78,39,-0.6959478885088008,1116,False), Edge(76,49,0.9709622470027257,1118,True), Edge(58,37,0.10778315608448508,1120,False), Edge(112,39,0.15474225915109896,3168,False), Edge(59,36,-0.6875440737399712,1122,True), Edge(75,36,-0.8545184101908228,1124,False), Edge(40,54,0.4547184589695824,3176,False), Edge(58,36,-0.5232040357664516,1129,False), Edge(59,36,-0.2347818349668831,1131,False), Edge(110,36,0.5689533297934748,3186,False), Edge(28,49,0.7433195201510598,3188,False), Edge(80,37,0.12646518139423457,1143,False), Edge(43,54,0.02095269324182525,1146,False), Edge(77,36,-0.13608537042692426,1156,True), Edge(59,48,0.98682736526802,1157,True), Edge(33,36,-0.07972507782170624,1160,False), Edge(26,36,-0.6864186564566612,1166,False), Edge(76,77,0.7133368701083049,1178,False), Edge(32,37,-0.5516753124265643,1190,False), Edge(60,36,-0.0036955812036809643,1191,False), Edge(110,54,-0.8556207055581007,3240,False), Edge(99,36,0.4522786332566828,3244,False), Edge(109,39,0.31121082366373054,3268,False), Edge(76,54,0.29734044473645604,1223,False), Edge(34,39,0.11493609280602635,1227,True), Edge(111,36,-0.2466762726062015,3275,False), Edge(79,78,-0.39972968893878313,1234,True), Edge(77,37,-0.3761706947316179,1245,False), Edge(44,48,0.6242975464990579,1247,False), Edge(112,36,-0.3735695828627672,3298,False), Edge(61,37,-0.8918878207459542,1251,False), Edge(42,78,0.15624276814445826,1254,True), Edge(113,78,-0.48205822377746,3307,False), Edge(113,36,-0.13608537042692426,3319,False), Edge(57,49,0.8417957701447749,1285,True), Edge(45,49,-0.6523281099821265,1296,False), Edge(81,39,0.28088403047692934,1307,False), Edge(60,39,-0.16843477841414645,1308,True), Edge(114,39,0.589512129302477,3361,False), Edge(61,48,0.8603991723606321,1315,True), Edge(114,87,0.23295866897799922,3363,False), Edge(26,36,0.8770136442518929,1318,False), Edge(115,54,0.1951007245550983,3367,False), Edge(62,81,-0.09699215384744653,1325,False), Edge(65,99,0.9811165929509954,3381,False), Edge(115,54,0.7372792819456198,3387,False), Edge(61,36,0.5974955150106886,3401,False), Edge(116,36,0.9844383962496319,3439,False), Edge(81,37,-0.6413515373899386,1395,False), Edge(82,37,0.5268319498165015,1421,False), Edge(62,37,-0.3570162942374415,1423,False), Edge(75,77,-0.5307110096314154,1432,False), Edge(117,87,0.3970083915245821,3488,False), Edge(100,39,0.28224346991931526,3565,True), Edge(22,77,0.9422907813758393,3568,False), Edge(83,49,0.3311963506327551,1527,False), Edge(83,36,0.7465021704738459,1543,False), Edge(16,87,0.24224367350706866,3592,False), Edge(84,37,0.5057481299313846,1549,False), Edge(84,36,-0.5692669168656832,1559,False), Edge(98,37,0.9870987102000552,3636,True), Edge(83,48,0.16296763735451303,1592,False), Edge(52,39,0.3355342754013755,1594,False), Edge(81,36,0.7042535162522163,1627,False), Edge(85,39,-0.38323813134329,1649,False), Edge(85,81,-0.25511575372551953,1653,False), Edge(85,36,-0.3735695828627672,1668,False), Edge(85,48,0.812894073355199,1691,False), Edge(86,37,-0.7593877351301184,1697,False), Edge(68,49,0.2942468328234755,1720,False), Edge(86,36,0.05925037023826052,1721,False), Edge(87,36,0.9917329972373357,1740,True), Edge(87,78,-0.39972968893878313,1744,False), Edge(27,87,0.5783664228546296,1793,False), Edge(40,78,0.13287445951599386,1798,False), Edge(65,81,0.7255972240388604,1813,False), Edge(88,37,-0.5032477120999719,1834,False), Edge(88,36,0.03094818394171517,1836,True), Edge(89,78,-0.11848825291342213,1846,True), Edge(89,54,0.952673298008416,1852,True), Edge(87,39,-0.8186732830398622,1855,False), Edge(60,54,0.6460908521386974,1897,False), Edge(79,48,-0.9945280974270221,1899,False), Edge(90,39,0.9425284113836925,1905,False), Edge(43,49,0.9773998934653928,1923,False), Edge(91,36,0.6769775276258676,1928,True), Edge(34,81,-0.3635390032185555,1933,False), Edge(90,49,0.44760706183174626,1950,False), Edge(90,36,0.14423856701348314,1962,False), Edge(92,49,0.2364405843527535,1983,False), Edge(24,78,0.6580511038327919,1992,False), Edge(90,37,0.5294891371403418,2002,False), Edge(58,49,-0.5975971020552444,2041,False), Edge(71,99,0.11873118475853706,3672,False), Edge(19,118,1.0,3695,False), Edge(118,54,0.07432979317369237,3695,False), Edge(48,119,1.0,3709,False), Edge(119,39,0.9735427239600714,3709,False)],
        [Edge(16,36,1,0,True), Edge(50,48,0.9679691697549194,2049,False), Edge(91,37,0.12967116897126774,2051,False), Edge(91,36,0.47822390502148004,2074,False), Edge(92,48,0.8853208447362337,2076,False), Edge(37,36,0.12307851440160733,46,True), Edge(93,78,0.5865861224878542,2132,True), Edge(37,36,1,97,True), Edge(92,39,0.24952337756836362,2152,False), Edge(93,37,0.7457428693554302,2163,False), Edge(91,87,0.9357686206976068,2171,False), Edge(93,48,0.42866970280661376,2177,False), Edge(26,48,0.10894550119257107,2200,False), Edge(94,87,0.23295866897799922,2201,True), Edge(37,36,0.2555094646175703,157,True), Edge(34,54,-0.07010372955148392,2205,False), Edge(93,48,-0.4523442394355155,2214,False), Edge(94,54,0.3271749339290806,2223,False), Edge(69,39,0.7432813763712653,2234,False), Edge(27,49,-0.1232948311572819,2240,True), Edge(94,54,-0.5817347848500549,2245,False), Edge(69,48,0.10750034172974532,2257,False), Edge(71,49,0.4337521845774851,2263,False), Edge(95,78,-0.39972968893878313,2264,False), Edge(89,48,0.12249921624037752,2286,False), Edge(37,36,1,244,True), Edge(94,78,-0.9044972991167137,2300,False), Edge(37,36,0.42676425202570956,259,False), Edge(63,54,0.6431584439039271,2307,False), Edge(37,36,0.9272693056268737,268,False), Edge(65,37,0.6929162324590197,2317,False), Edge(95,48,0.2905311220101552,2318,False), Edge(95,49,-0.5109531612261211,2342,False), Edge(52,87,0.19795411872734736,2344,False), Edge(37,36,0.6965624474859198,298,True), Edge(38,36,-0.7249711491713702,302,False), Edge(96,39,-0.47479626650104567,2351,False), Edge(16,39,0.7597861335895142,320,False), Edge(63,36,-0.1310605615076461,322,False), Edge(61,81,-0.36169608461516023,2370,False), Edge(97,49,0.37407567433336975,2375,False), Edge(22,36,-0.9255772972135201,333,False), Edge(63,36,0.8192624571415981,336,False), Edge(38,37,0.5616260727066895,338,True), Edge(23,54,0.7845467222343699,346,True), Edge(24,48,0.2610689625089182,349,False), Edge(39,36,-0.4496946419832797,353,True), Edge(64,36,0.028812382922944524,355,False), Edge(96,36,-0.9181837764361287,2404,False), Edge(65,36,0.03717027728480371,359,False), Edge(6,36,0.17397474248127132,374,True), Edge(40,39,0.7486775375863762,385,True), Edge(96,36,0.03094818394171517,2434,False), Edge(41,36,-0.36531132475233763,387,True), Edge(64,54,0.7845467222343699,391,True), Edge(47,36,0.9272693056268737,395,True), Edge(40,36,-0.021039102074985605,398,True), Edge(66,36,-0.1411865355904658,401,False), Edge(40,36,-0.22727225089644887,403,True), Edge(12,54,-0.6254203702321586,407,True), Edge(41,36,0.6360675157869415,416,False), Edge(98,78,0.3929434251033561,2466,False), Edge(42,39,-0.29542648698059004,2468,False), Edge(29,39,-0.9479014055095569,424,True), Edge(27,37,0.8088518430566118,428,True), Edge(40,36,-0.021039102074985605,430,True), Edge(48,36,0.6903649795829547,439,False), Edge(6,37,0.9967530259749053,440,False), Edge(10,39,0.7732261879326431,441,True), Edge(48,36,-0.40669043820716033,443,False), Edge(33,37,-0.08287174811297371,447,True), Edge(97,78,-0.5804240053638561,2498,False), Edge(66,37,0.5029298588831674,451,False), Edge(79,87,0.43102974637419,2500,False), Edge(97,39,0.28088403047692934,2504,False), Edge(99,77,0.19519840928031162,2506,False), Edge(42,36,-0.7014444413534948,459,True), Edge(19,54,0.07432979317369237,462,False), Edge(49,36,-0.4238627588717503,463,True), Edge(67,37,-0.123591026422887,464,False), Edge(98,54,0.7221856675217535,2512,True), Edge(23,48,0.09755292303815444,2515,False), Edge(48,36,-0.9432525189286147,479,False), Edge(0,37,0.8692034596775957,483,False), Edge(50,37,0.2976123467309044,487,True), Edge(73,77,0.7135266761826092,2536,False), Edge(100,36,-0.5821594446889009,2539,False), Edge(6,37,0.2899804310606785,492,True), Edge(59,49,-0.39748565960729776,2544,False), Edge(98,36,0.03094818394171517,2545,False), Edge(43,36,-0.8459905063536599,499,True), Edge(101,54,0.20444200602000495,2555,False), Edge(41,36,0.2154248654415527,509,False), Edge(6,39,0.8386498259372221,520,False), Edge(49,36,-0.6158664189451919,525,True), Edge(102,37,0.18647850057134585,2574,False), Edge(28,36,-0.3766378384817164,531,False), Edge(50,37,0.5757705304464205,532,True), Edge(19,37,0.6204277756795049,535,False), Edge(34,37,0.5455689424840654,2587,False), Edge(40,48,0.3841901115445554,543,False), Edge(44,36,-0.318871195041148,546,True), Edge(49,39,0.15474225915109896,547,True), Edge(51,39,0.3044822132751166,551,False), Edge(46,37,0.18647850057134585,553,True), Edge(38,77,0.7598903060247231,2603,False), Edge(99,54,0.7372792819456198,2605,False), Edge(73,99,0.9571134675934222,2607,False), Edge(32,39,0.6229121163827249,577,False), Edge(43,37,0.49205810720018306,578,True), Edge(51,37,0.8088518430566118,579,True), Edge(67,54,0.1951007245550983,2629,True), Edge(68,36,-0.021039102074985605,586,True), Edge(3,39,0.9514983128252149,589,True), Edge(100,36,-0.6158664189451919,2639,False), Edge(40,49,0.3673087034485545,593,False), Edge(30,36,-0.21084161144019475,594,False), Edge(48,36,-0.5257909877238525,595,True), Edge(47,48,0.6584321854037208,603,True), Edge(68,39,-0.16843477841414645,604,False), Edge(101,49,0.5638759426360338,2666,False), Edge(103,49,0.9709622470027257,2676,False), Edge(52,37,0.6217805232551452,633,True), Edge(64,37,-0.7018093072249494,636,False), Edge(27,54,-0.06472412668119976,643,False), Edge(69,36,0.6769775276258676,644,True), Edge(51,48,0.1442134657829075,646,False), Edge(41,54,0.5980293099361317,647,True), Edge(69,49,-0.7179995372782297,653,False), Edge(48,36,0.9559533885581981,668,False), Edge(102,36,-0.7638533973570822,2724,False), Edge(45,36,-0.07895095253087048,684,True), Edge(104,78,0.5865861224878542,2749,False), Edge(69,54,0.3096801068893904,705,False), Edge(53,37,0.12341540956682451,708,False), Edge(103,39,0.013488639246391099,2760,False), Edge(27,36,-0.4947984123112614,718,False), Edge(49,37,-0.20213795065088713,724,True), Edge(69,37,0.06334118461519522,729,False), Edge(48,39,0.9735427239600714,733,False), Edge(104,36,0.6629961922888061,2783,False), Edge(42,48,-0.5357384674653098,740,False), Edge(50,36,-0.9189911588578752,741,False), Edge(15,48,0.0609814731996825,746,True), Edge(37,39,0.8510031486544098,2796,False), Edge(105,36,0.8770136442518929,2797,False), Edge(69,54,0.6512900529302501,750,True), Edge(106,36,-0.912799608059162,2801,False), Edge(46,36,0.9425030441229787,757,True), Edge(70,39,-0.7241689671198921,758,False), Edge(87,37,0.48270368382170215,2812,False), Edge(70,37,0.9352008356887924,776,False), Edge(50,36,-0.31780587598548293,778,False), Edge(51,49,-0.6542842758367244,791,False), Edge(105,37,-0.16156693597102834,2850,False), Edge(80,77,0.5518855436125463,2863,False), Edge(54,36,0.12434244685395823,820,True), Edge(51,36,-0.5615210108751827,825,True), Edge(71,37,-0.3166746999466068,831,False), Edge(88,99,0.6883351794749062,2880,False), Edge(31,37,0.48024458065874875,835,False), Edge(46,39,0.6881431340485101,844,True), Edge(72,36,0.6842896406971593,856,False), Edge(68,54,-0.496908389346427,863,False), Edge(86,87,-0.1651553363950351,2916,False), Edge(107,36,0.5689533297934748,2917,True), Edge(21,54,-0.7042177540003556,872,False), Edge(20,36,0.32235576244518316,875,False), Edge(71,36,-0.911729055626187,892,False), Edge(71,36,0.6799222938352465,896,True), Edge(73,36,0.9703340759346426,898,False), Edge(18,37,0.08671669617684874,900,True), Edge(70,36,0.14386466895588956,906,False), Edge(55,36,0.1028222288030296,907,True), Edge(42,37,0.24933978575832327,911,False), Edge(64,49,0.6851991968516669,922,False), Edge(62,54,0.8941170676594443,923,False), Edge(54,39,-0.18923272502332633,926,True), Edge(52,49,0.5782329042878998,931,False), Edge(49,77,0.03369200703881248,2984,False), Edge(70,49,0.9414942957082036,937,True), Edge(72,49,0.7924538330731987,940,False), Edge(28,39,0.589512129302477,943,False), Edge(72,36,0.8930783645544096,944,True), Edge(106,54,0.9824176035915613,2992,False), Edge(47,49,0.1899145862261612,947,True), Edge(21,37,-0.16156693597102834,948,False), Edge(34,49,-0.5809061845330319,955,True), Edge(73,36,0.9703340759346426,959,False), Edge(107,54,0.5267100171182313,3008,False), Edge(55,36,0.5258863264859874,961,False), Edge(72,36,1,964,False), Edge(66,49,0.05780241350392856,3024,False), Edge(72,37,0.2976123467309044,978,True), Edge(40,99,0.3013978719526301,3028,False), Edge(108,39,0.04779982813038397,3029,False), Edge(28,48,0.63082093115516,983,False), Edge(74,39,0.005726280127722916,989,False), Edge(106,36,0.6829265888515275,3039,False), Edge(74,39,0.0515825932676095,1001,True), Edge(109,54,0.02095269324182525,3058,False), Edge(7,39,0.3161055749535986,1013,False), Edge(44,78,0.5727716328391782,3061,False), Edge(34,48,0.7197811690610363,1015,True), Edge(1,78,0.6258734448444967,3066,False), Edge(108,36,0.596270410300471,3067,False), Edge(75,37,0.5294891371403418,1023,True), Edge(109,37,0.8517384667636347,3072,False), Edge(74,54,-0.9883092020061461,1025,False), Edge(74,36,-0.20351815441442156,1029,False), Edge(76,99,0.15137250773277022,3079,False), Edge(108,37,0.12341540956682451,3080,False), Edge(23,36,0.8728329585427566,1035,False), Edge(71,37,-0.9303942608921612,1036,True), Edge(106,77,-0.010044462550975464,3088,False), Edge(56,87,0.4770027831024106,3092,False), Edge(107,36,-0.21084161144019475,3093,False), Edge(90,77,0.14041706805172716,3096,False), Edge(53,54,-0.48833909782057505,1055,False), Edge(109,77,-0.8512444232608283,3108,False), Edge(56,36,0.9880295265779211,1062,False), Edge(106,36,0.5402867161899334,3110,False), Edge(75,36,-0.2466762726062015,1064,False), Edge(76,36,0.5616274901734499,1067,False), Edge(57,36,-0.6440467910393914,1071,True), Edge(43,48,-0.948844064224063,1073,False), Edge(110,49,-0.6523281099821265,3125,False), Edge(56,48,-0.4749878742598257,1078,True), Edge(88,39,-0.9779891802416121,3128,False), Edge(76,48,-0.7922756988590154,1081,True), Edge(71,54,0.6151253175494457,1083,True), Edge(58,36,-0.888586056035876,1085,False), Edge(111,48,0.4387163410835342,3144,False), Edge(77,39,0.1999769856424527,1103,False), Edge(78,39,-0.6959478885088008,1116,False), Edge(76,49,0.9709622470027257,1118,True), Edge(58,37,0.10778315608448508,1120,False), Edge(112,39,0.15474225915109896,3168,False), Edge(59,36,-0.6875440737399712,1122,True), Edge(75,36,-0.8545184101908228,1124,False), Edge(40,54,0.4547184589695824,3176,False), Edge(58,36,-0.5232040357664516,1129,False), Edge(59,36,-0.2347818349668831,1131,False), Edge(110,36,0.5689533297934748,3186,False), Edge(28,49,0.7433195201510598,3188,False), Edge(80,37,0.12646518139423457,1143,False), Edge(43,54,0.02095269324182525,1146,False), Edge(77,36,-0.13608537042692426,1156,True), Edge(59,48,0.98682736526802,1157,True), Edge(33,36,-0.07972507782170624,1160,False), Edge(26,36,-0.6864186564566612,1166,False), Edge(76,77,0.7133368701083049,1178,False), Edge(32,37,-0.5516753124265643,1190,False), Edge(60,36,-0.0036955812036809643,1191,False), Edge(110,54,0.896482669097791,3240,False), Edge(99,36,0.4522786332566828,3244,False), Edge(109,39,0.31121082366373054,3268,False), Edge(76,54,0.29734044473645604,1223,False), Edge(34,39,0.11493609280602635,1227,True), Edge(111,36,-0.2466762726062015,3275,False), Edge(79,78,-0.39972968893878313,1234,True), Edge(77,37,-0.3761706947316179,1245,False), Edge(44,48,0.6242975464990579,1247,False), Edge(112,36,-0.3735695828627672,3298,False), Edge(61,37,-0.8918878207459542,1251,False), Edge(42,78,0.15624276814445826,1254,True), Edge(113,78,-0.48205822377746,3307,False), Edge(113,36,-0.13608537042692426,3319,False), Edge(57,49,0.8417957701447749,1285,True), Edge(45,49,-0.6523281099821265,1296,False), Edge(81,39,0.28088403047692934,1307,False), Edge(60,39,-0.16843477841414645,1308,True), Edge(114,39,0.589512129302477,3361,False), Edge(61,48,0.8603991723606321,1315,True), Edge(114,87,0.23295866897799922,3363,False), Edge(26,36,0.8770136442518929,1318,False), Edge(115,54,0.1951007245550983,3367,False), Edge(62,81,-0.09699215384744653,1325,False), Edge(65,99,0.9811165929509954,3381,False), Edge(115,54,0.7372792819456198,3387,False), Edge(61,36,0.5974955150106886,3401,False), Edge(116,36,0.9844383962496319,3439,False), Edge(81,37,-0.6413515373899386,1395,False), Edge(82,37,0.5268319498165015,1421,False), Edge(62,37,-0.3570162942374415,1423,False), Edge(75,77,-0.5307110096314154,1432,False), Edge(117,87,0.3970083915245821,3488,False), Edge(100,39,0.28224346991931526,3565,True), Edge(22,77,0.9422907813758393,3568,False), Edge(83,49,0.3311963506327551,1527,False), Edge(83,36,0.7465021704738459,1543,False), Edge(16,87,0.24224367350706866,3592,False), Edge(84,37,0.5057481299313846,1549,False), Edge(84,36,-0.5692669168656832,1559,False), Edge(98,37,0.8392957128479024,3636,False), Edge(83,48,0.16296763735451303,1592,False), Edge(52,39,0.3355342754013755,1594,False), Edge(81,36,0.7042535162522163,1627,False), Edge(85,39,-0.38323813134329,1649,False), Edge(85,81,-0.25511575372551953,1653,False), Edge(85,36,-0.3735695828627672,1668,False), Edge(85,48,0.812894073355199,1691,False), Edge(86,37,-0.7593877351301184,1697,False), Edge(68,49,0.2942468328234755,1720,False), Edge(86,36,0.05925037023826052,1721,False), Edge(87,36,0.9917329972373357,1740,True), Edge(87,78,-0.39972968893878313,1744,False), Edge(27,87,0.5783664228546296,1793,False), Edge(40,78,0.13287445951599386,1798,False), Edge(65,81,0.7255972240388604,1813,False), Edge(88,37,-0.5032477120999719,1834,False), Edge(88,36,0.03094818394171517,1836,True), Edge(89,78,-0.11848825291342213,1846,True), Edge(89,54,0.952673298008416,1852,True), Edge(87,39,-0.8186732830398622,1855,False), Edge(60,54,0.6460908521386974,1897,False), Edge(79,48,-0.9945280974270221,1899,False), Edge(90,39,0.9425284113836925,1905,False), Edge(43,49,0.4835547891533789,1923,False), Edge(91,36,0.6769775276258676,1928,True), Edge(34,81,-0.3635390032185555,1933,False), Edge(90,49,0.44760706183174626,1950,False), Edge(90,36,0.14423856701348314,1962,False), Edge(92,49,0.2364405843527535,1983,False), Edge(24,78,0.6580511038327919,1992,False), Edge(90,37,0.5294891371403418,2002,False), Edge(58,49,-0.5975971020552444,2041,False)]
            ]
    NEAT(1000,True,printMod=True,preGene=preGene)