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
    preGene = [[Edge(10,43,1,0,True), Edge(21,44,-0.712753581079218,513,False), Edge(92,44,-0.3984556457221671,1028,False), Edge(16,44,-0.7158326491134126,1031,False), Edge(12,45,-0.45532211060143757,521,False), Edge(44,43,0.5951031171078358,13,True), Edge(59,44,-0.7242806917290314,1039,False), Edge(40,64,-0.14931449996981083,2064,False), Edge(34,64,-0.08642955930910046,1040,True), Edge(5,43,-0.15571568068013386,1043,True), Edge(60,43,-0.14866448788008446,1044,False), Edge(20,44,0.6606349094689965,541,False), Edge(99,66,0.7999261422947195,1567,True), Edge(50,43,-0.44305564524540286,546,True), Edge(27,45,0.806106338577739,1571,False), Edge(61,43,0.9748625554773562,1060,False), Edge(1,43,-0.7527345119736506,550,True), Edge(106,44,-0.6392678045660583,2092,False), Edge(54,43,-0.11189756529615491,557,False), Edge(52,66,0.9083815709750824,561,True), Edge(67,44,0.8816357374128363,1073,False), Edge(49,45,-0.34132753803268057,1077,False), Edge(93,66,-0.3151839211088576,1080,True), Edge(38,44,-0.9109665088895227,569,True), Edge(8,43,-0.7757532163639007,571,True), Edge(52,64,-0.7965033170077433,1085,False), Edge(66,44,-0.09575027003061054,575,False), Edge(72,68,0.3999579115338354,1087,False), Edge(44,43,-0.4317793757195256,70,True), Edge(1,45,0.8797853789189662,583,True), Edge(62,64,-0.20201458622422486,584,False), Edge(79,62,0.9265393486598377,2121,False), Edge(100,64,0.5005441737288514,1608,False), Edge(2,45,0.9343228181068743,587,False), Edge(6,45,-0.18005422245269354,1100,False), Edge(99,43,-0.6213998213090548,1613,False), Edge(39,44,-0.4490057484779095,2126,False), Edge(51,43,0.14143931829870393,591,False), Edge(55,45,-0.21414234284145794,594,True), Edge(54,45,0.2728280027781018,595,False), Edge(102,62,0.6596957677084847,1848,False), Edge(93,44,0.2665477609226805,1109,False), Edge(96,44,-0.3173150761081376,1340,False), Edge(67,45,0.03405097425793757,599,False), Edge(44,43,0.3491233778129299,88,True), Edge(93,68,-0.05032862168116625,1111,False), Edge(91,45,-0.8866437356749399,1115,False), Edge(94,66,-0.061169041365882526,1116,True), Edge(28,64,0.024410455268009335,606,True), Edge(34,43,-0.2569957085012451,608,False), Edge(24,66,0.3145928084083378,609,True), Edge(46,82,-0.1553341403688986,611,False), Edge(107,62,-0.599710185817339,2148,False), Edge(67,43,-0.061312073256672095,613,False), Edge(37,82,-0.5322068959797643,614,True), Edge(66,43,0.19262571058070832,618,False), Edge(85,44,-0.5803638263645556,2155,True), Edge(67,64,-0.8595158391217559,1643,False), Edge(49,66,0.15298594948946942,1854,False), Edge(99,44,-0.5044187315491513,2162,False), Edge(88,66,0.37814878116729167,626,False), Edge(47,62,-0.6280851548137902,627,False), Edge(93,44,-0.7334522390773195,1144,False), Edge(89,43,0.04180940716665482,633,False), Edge(39,64,-0.6916582813337149,1147,False), Edge(73,62,-0.3822522717277843,1149,True), Edge(37,44,-0.46711867618798086,639,True), Edge(67,43,-0.6609348240912036,643,True), Edge(51,43,0.00010277718251661128,646,False), Edge(72,45,-0.5012313743845376,2185,False), Edge(44,43,0.5826631500159063,139,True), Edge(66,43,0.1065151356363987,652,True), Edge(100,66,0.8615144250632631,1686,True), Edge(52,43,0.692931694111774,663,True), Edge(58,45,-0.28760341373751497,1688,False), Edge(55,44,-0.9601568046920232,667,True), Edge(68,64,-0.7456923892340406,673,True), Edge(94,45,0.42294484858960857,1186,False), Edge(36,43,-0.08136871363530962,675,True), Edge(61,44,0.7979442757831949,1187,False), Edge(41,45,-0.784802711191503,677,True), Edge(108,44,-0.7466915684133575,2215,False), Edge(59,64,-0.4689442360589331,1192,False), Edge(101,66,0.9083815709750824,1705,False), Edge(74,44,0.1700104343486435,1194,False), Edge(100,66,-0.236016533393959,1709,True), Edge(95,43,-0.8223045986425144,1198,True), Edge(94,43,0.3300109217260765,1200,True), Edge(4,45,0.9580838916644596,689,False), Edge(44,43,0.5694511443485124,178,True), Edge(107,43,-0.061312073256672095,2227,False), Edge(88,68,-0.10660924105714353,691,False), Edge(53,62,-0.2951107660869732,1712,False), Edge(94,82,0.8410522246878402,1713,False), Edge(107,44,0.16567989660393923,2231,False), Edge(68,43,-0.2647625467320156,695,True), Edge(26,64,0.12014566788710934,2233,False), Edge(49,44,0.4088378666046466,696,False), Edge(56,43,-0.6406562219540768,699,False), Edge(90,45,0.4763267228424224,1208,False), Edge(107,66,0.15298594948946942,2237,False), Edge(58,66,0.8615144250632631,1211,False), Edge(107,44,-0.42039453885024103,2239,False), Edge(3,45,0.6125718641739826,1213,False), Edge(29,45,-0.5087761186536803,705,True), Edge(75,43,0.37882477440769735,1215,False), Edge(44,43,0.4829283989918085,199,False), Edge(68,43,0.9721130430536122,712,True), Edge(57,82,0.08194224338982292,2250,True), Edge(101,43,-0.8535350513599587,1739,False), Edge(52,43,0.692931694111774,721,False), Edge(89,64,0.9819091855863911,2258,False), Edge(94,43,-0.8535350513599587,1233,True), Edge(94,66,0.29153206131223386,1235,False), Edge(101,45,-0.39078785450671516,1746,False), Edge(32,45,0.663877819580396,726,False), Edge(108,45,-0.6308441682949901,2265,False), Edge(80,43,-0.11267214244516821,732,True), Edge(90,62,-0.599710185817339,734,False), Edge(28,45,-0.36581739964148396,735,True), Edge(88,45,-0.3189635275722267,1253,False), Edge(67,66,-0.756448687753535,2278,False), Edge(84,45,0.44380484453692914,742,True), Edge(93,43,-0.08490261262719279,1256,False), Edge(96,45,0.4954967933471761,1260,False), Edge(89,43,-0.4068828939972169,749,False), Edge(4,64,-0.10073034029334105,750,False), Edge(63,68,-0.8278477631146659,751,True), Edge(44,43,0.9874571888311285,241,True), Edge(89,43,0.8509644000405336,753,True), Edge(34,66,-0.5658155934238736,1779,False), Edge(68,45,-0.7593735356031253,756,False), Edge(22,44,-0.49133809902822434,757,False), Edge(53,44,-0.8430146571382631,758,False), Edge(90,43,0.11423511875568937,760,False), Edge(109,45,-0.9570011941983554,2298,False), Edge(102,43,-0.6250514581962581,1786,False), Edge(96,43,0.663112844668934,1278,False), Edge(54,44,0.743571852280968,767,False), Edge(37,66,0.7472090003735381,773,False), Edge(109,66,-0.6925890631326133,2311,False), Edge(53,44,-0.12256795355510364,778,True), Edge(108,45,0.47372131601779865,2318,True), Edge(95,82,0.4365887210551027,1295,False), Edge(109,45,-0.6650394582663557,2320,False), Edge(36,62,0.9797884980275511,1296,False), Edge(91,82,0.5234277747749354,789,False), Edge(67,44,0.2802990599596351,1303,False), Edge(96,82,0.7018123093092725,1308,True), Edge(34,45,-0.9466160478967025,797,False), Edge(102,64,0.5802005300420889,1822,False), Edge(38,43,-0.6647797624272571,1312,True), Edge(65,45,-0.17109781093940457,804,False), Edge(110,45,0.47372131601779865,2342,False), Edge(68,43,0.8320126821001976,807,True), Edge(0,43,-0.6609348240912036,808,False), Edge(58,44,0.7993640568165694,1319,True), Edge(57,66,0.4390498126586344,1320,True), Edge(94,82,-0.028046931737249592,1831,False), Edge(68,45,0.27233317363693343,812,True), Edge(55,43,-0.7597821046368711,813,False), Edge(28,43,-0.5475180495352123,302,True), Edge(69,43,-0.17414989748242515,814,True), Edge(62,45,0.44380484453692914,304,True), Edge(2,43,-0.09266398718952162,817,True), Edge(76,43,-0.719199813410665,1326,False), Edge(13,44,0.16852120712274266,307,False), Edge(102,43,0.3300109217260765,1840,False), Edge(70,43,0.6678264940348166,821,True), Edge(94,44,0.9055130374067877,1333,False), Edge(62,44,0.6108045495027228,312,False), Edge(96,62,-0.02602500355027426,2361,False), Edge(49,44,0.5359946794582107,313,True), Edge(26,45,0.36517649284782094,314,False), Edge(71,66,-0.7600334304253011,2364,False), Edge(45,44,-0.6749450218803892,315,False), Edge(25,43,0.8003245490989297,318,True), Edge(36,82,0.4763994806013727,1336,False), Edge(8,62,0.10671957382174369,320,False), Edge(63,44,0.3806051545524147,321,False), Edge(25,44,0.31348075752476046,322,True), Edge(70,45,0.2130776470808109,832,False), Edge(57,44,0.6604283431806024,834,True), Edge(50,44,0.5359946794582107,325,True), Edge(19,45,-0.9207669605632263,836,False), Edge(46,66,-0.5402310219954962,327,False), Edge(79,43,0.521680096287938,328,False), Edge(57,45,0.5999593672962404,838,True), Edge(97,44,-0.7277098902645456,1345,False), Edge(72,66,0.7775094302412515,1347,False), Edge(76,62,-0.2622534297242096,1866,False), Edge(71,43,0.7194011616056086,845,True), Edge(45,43,-0.9367781546143004,334,True), Edge(46,44,0.05627187571031533,335,False), Edge(103,43,-0.024549401907954405,1871,False), Edge(60,45,-0.9075907772024079,337,False), Edge(13,82,-0.6427909363608182,338,False), Edge(13,45,0.44779731104890597,339,False), Edge(75,62,-0.5936691356590538,849,False), Edge(53,68,0.498164400641403,1361,False), Edge(36,45,-0.019019262905032974,342,True), Edge(28,44,-0.06784561366613251,343,True), Edge(68,44,0.08423009334143461,1363,False), Edge(20,66,0.3315531881952558,345,True), Edge(76,43,-0.20084648141681916,1368,True), Edge(40,45,-0.08733862607351761,859,False), Edge(103,62,0.319491449724574,1878,False), Edge(103,43,0.11423511875568937,1884,False), Edge(79,45,0.31011229267565943,351,True), Edge(64,43,0.09266699173213033,355,True), Edge(102,45,0.19933677241718928,1891,False), Edge(63,43,-0.5322439970672019,358,True), Edge(85,43,-0.6791723894672579,359,True), Edge(97,66,-0.799660777172235,1382,False), Edge(42,44,0.33453828833952937,361,True), Edge(11,66,-0.9364310287650945,362,False), Edge(27,43,0.11041161887496531,364,True), Edge(81,43,-0.48512295255867066,365,True), Edge(34,66,-0.5658155934238736,1900,False), Edge(85,45,0.8601781347628228,368,False), Edge(104,45,0.47372131601779865,1905,True), Edge(82,43,0.1909472653652291,370,False), Edge(90,43,0.2935527787634755,883,False), Edge(81,44,0.22816786279487244,372,False), Edge(54,62,0.11423511875568937,377,True), Edge(49,64,0.5005441737288514,1403,False), Edge(52,68,0.5105448032973547,892,True), Edge(98,43,-0.3756937858706264,1405,False), Edge(64,44,-0.9308612057500862,382,False), Edge(7,43,-0.7097353472323571,383,True), Edge(70,45,0.2563919412470079,894,True), Edge(54,43,-0.6179855612701934,896,False), Edge(37,64,-0.7850054180787145,1920,False), Edge(49,43,0.33625217576472477,388,True), Edge(76,43,-0.8223045986425144,1414,True), Edge(38,45,0.42294484858960857,903,True), Edge(46,43,0.7685685187134161,394,True), Edge(32,43,0.39268373828408487,396,True), Edge(45,43,-0.07759295731774696,399,True), Edge(85,43,0.9164844262963596,402,False), Edge(46,43,-0.8077736831704678,404,True), Edge(47,43,-0.8607197561401951,406,True), Edge(63,66,-0.07145390920939354,410,True), Edge(38,68,-0.9985898036045118,922,False), Edge(77,44,-0.4672960021894921,1436,False), Edge(22,45,-0.5676178283239404,413,True), Edge(33,62,0.5961712160818349,1950,False), Edge(65,64,0.11913049113969643,416,False), Edge(86,45,-0.9514644746068393,417,False), Edge(60,62,-0.4308991292425721,418,False), Edge(78,64,0.7135011486885241,1440,False), Edge(13,43,-0.3150032399572693,420,True), Edge(91,62,0.5355374920539491,932,False), Edge(58,43,0.3786100821915166,935,True), Edge(39,43,-0.3485005772107159,939,False), Edge(60,44,0.8801518461486812,428,True), Edge(71,44,-0.012213979023311339,940,False), Edge(80,45,-0.7905871515228957,430,False), Edge(58,68,0.2214123661637648,1965,False), Edge(55,66,0.8180335835956376,432,False), Edge(51,43,0.04697766191345032,433,True), Edge(8,45,0.7440236123580743,435,True), Edge(50,44,0.007545548730783613,436,True), Edge(90,82,-0.9322462111762149,949,False), Edge(91,62,-0.8530147925206462,951,False), Edge(19,45,1,953,True), Edge(82,45,-0.9262262928941292,442,False), Edge(92,66,0.1709089383376554,955,False), Edge(82,45,-0.9262262928941292,444,False), Edge(105,66,0.2653183778539052,1980,False), Edge(104,45,0.36517649284782094,1982,False), Edge(48,43,-0.053043942192943616,447,True), Edge(52,44,-0.992296761553189,448,True), Edge(53,64,-0.6386368540182108,960,False), Edge(83,66,-0.2205588948988193,451,False), Edge(59,43,-0.42537727348841603,967,True), Edge(98,43,-0.8223045986425144,1480,False), Edge(33,44,-0.025044937740689388,969,False), Edge(104,66,-0.8001308326017933,1994,False), Edge(0,45,0.3645069602316928,461,True), Edge(105,45,-0.21414234284145794,1997,False), Edge(55,82,-0.20520546850257415,464,False), Edge(25,64,0.319060197502014,979,False), Edge(106,66,-0.9844795443630532,2008,False), Edge(83,43,0.2712282394739929,473,False), Edge(87,43,-0.3982579307394478,481,True), Edge(52,43,0.7780783399860491,483,False), Edge(78,43,0.2520974399361713,2026,False), Edge(53,44,0.24083053459835768,492,True), Edge(50,45,0.8010189726590116,1005,False), Edge(47,66,0.02082681924674179,1518,True), Edge(34,44,0.5166195352196041,498,False), Edge(72,64,0.5513093881948485,1011,False), Edge(91,44,-0.9216960535750749,1012,False), Edge(65,44,0.6427295688187327,501,False), Edge(53,43,1,502,True), Edge(99,68,0.5710940904405699,1523,False), Edge(72,43,-0.4416603744895764,1017,False), Edge(92,45,0.8371837407426066,1018,False), Edge(39,82,0.7338079842008087,2044,False), Edge(12,44,-0.7482366356210437,511,True), Edge(112,66,0.7472090003735381,2432,True), Edge(114,66,-0.3151839211088576,2691,False), Edge(113,44,-0.8520520852705058,2596,False), Edge(33,45,0.9923697797099749,2475,False), Edge(60,110,0.683568306414229,2383,True), Edge(46,45,0.2175649016083372,2423,False), Edge(111,66,-0.236016533393959,2392,False), Edge(112,44,-0.0008122129093959263,2494,False), Edge(112,115,1.0,2743,False), Edge(115,66,0.7472090003735381,2743,False), Edge(100,116,1.0,2795,False), Edge(116,66,0.8615144250632631,2795,False)]]
    NEAT(1000,True,printMod=True,preGene=preGene)