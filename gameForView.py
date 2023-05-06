from NEAT import *
import pygame,sys,random
from parameters import *
from sprite import *

if __name__ == "__main__":
    def forward(boxes,playerPos,nn):
        def check(n):
            if type(n) == Box:
                return 0.05*n.hp
            elif type(n) == Item:
                return 0.9
            else:
                return 0
        x = (*list(map(check,sum(boxes,[]))),playerPos/SCREEN_WIDTH)
        #x = (*list(map(check,boxes)),playerPos/SCREEN_WIDTH,nowRound*0.05,ballNumber*0.05)
        return nn.forward(*x)
    pygame.init()
    pygame.display.set_caption("SwipeBrickBreak")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    nowRound = 1
    boxes = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]
    bullets = []
    nowShooting = False
    playerPos = SCREEN_WIDTH/2
    startPos = SCREEN_HEIGHT - BULLET_SIZE/2
    delay = 0
    cnt = 0
    ballNumber = 1
    addBall = 0
    comeBackBall = 0
    moveToPos = playerPos
    nn = Topology(INPUT_NUM, OUTPUT_NUM)
    edges = [Edge(11,45,1,0,True), Edge(10,45,0.31820876699338674,324,False), Edge(46,45,1,133,True), Edge(24,46,0.7384731866626848,393,False), Edge(46,45,0.49999960326743736,298,False), Edge(48,45,0.5648062217784794,431,True), Edge(13,46,0.2679762511178482,404,False), Edge(46,45,0.9246861853044169,85,True), Edge(47,45,-0.2412986211688054,378,False), Edge(49,48,0.5131900545513307,484,False), Edge(49,45,0.638459716100725,486,False), Edge(4,46,-0.0718237351744459,550,False), Edge(3,45,-0.22741311076349652,490,False), Edge(5,45,0.8636664303916188,430,False), Edge(47,45,0.022223677944136933,398,False), Edge(40,45,0.8438401438462697,497,False), Edge(25,48,0.40145043460529506,468,False), Edge(48,45,0,500,False), Edge(24,45,0.5530154782604499,469,False), Edge(42,45,0.8279842878743203,533,False), Edge(46,45,0.12507532915507025,220,True)]
    edges = [Edge(10,43,1,0,True), Edge(21,44,0.46635958301507174,513,False), Edge(12,45,0.20269486658143765,521,False), Edge(44,43,0.5951031171078358,13,True), Edge(59,44,0.2696278801822791,1039,False), Edge(34,64,-0.08642955930910046,1040,True), Edge(5,43,-0.15571568068013386,1043,True), Edge(60,43,0.9384348543309147,1044,False), Edge(20,44,0.1055727080206903,541,False), Edge(50,43,-0.44305564524540286,546,True), Edge(61,43,-0.2779997578026634,1060,False), Edge(1,43,-0.7527345119736506,550,False), Edge(54,43,-0.14747963732640357,557,False), Edge(18,68,0.05322789992628718,571,False), Edge(66,44,0.056964857299666605,575,False), Edge(44,43,-0.4317793757195256,70,True), Edge(1,45,0.8797853789189662,583,True), Edge(6,45,-0.18005422245269354,1100,False), Edge(51,43,0.06186087136463292,591,False), Edge(55,45,-0.5901355354473239,594,False), Edge(54,45,0.2662022321428794,595,False), Edge(23,43,-0.1512518725284484,1109,True), Edge(67,45,-0.8412992589305388,599,False), Edge(44,43,0.3491233778129299,88,True), Edge(28,64,0.024410455268009335,606,False), Edge(34,43,-0.19472122607350295,608,False), Edge(24,66,0.3145928084083378,609,False), Edge(67,43,-0.1431751269767103,613,False), Edge(51,43,0.2571657348412155,614,True), Edge(66,43,-0.8934848643636013,618,False), Edge(47,62,-0.15009784639239432,627,False), Edge(73,62,-0.9216602776196796,1149,False), Edge(37,44,-0.3098478095613235,639,False), Edge(67,43,-0.6609348240912036,643,True), Edge(51,43,0.7133171733434189,646,False), Edge(44,43,0.5826631500159063,139,True), Edge(66,43,0.1065151356363987,652,False), Edge(52,43,-0.3713513709517382,663,True), Edge(55,44,-0.9601568046920232,667,True), Edge(68,64,-0.7456923892340406,673,True), Edge(36,43,-0.08136871363530962,675,True), Edge(61,44,0.8148588945438475,1187,False), Edge(41,45,0.5518327422278637,677,False), Edge(74,44,0.5525708706780708,1194,False), Edge(4,45,0.2511164516183517,689,False), Edge(44,43,0.5694511443485124,178,True), Edge(68,43,0.8320126821001976,695,False), Edge(49,44,-0.4532085332739728,696,False), Edge(56,43,-0.19441855001794783,699,False), Edge(58,66,0.17200655487379413,1211,False), Edge(3,45,0.6697010125710559,1213,False), Edge(75,43,0.2304752051080694,1215,False), Edge(29,45,-0.5087761186536803,705,True), Edge(44,43,0.4829283989918085,199,False), Edge(68,43,-0.6586623112535528,712,False), Edge(52,43,-0.5726435212213377,721,True), Edge(32,45,-0.2927556708092154,726,False), Edge(69,45,-0.311423015376632,732,False), Edge(28,45,-0.36581739964148396,735,True), Edge(4,64,-0.10073034029334105,750,False), Edge(63,68,-0.8278477631146659,751,True), Edge(44,43,0.9874571888311285,241,True), Edge(68,45,-0.6250754119397264,756,False), Edge(22,44,0.06564488626516174,757,False), Edge(53,44,-0.795627844342937,758,False), Edge(54,44,0.4476992911939144,767,False), Edge(53,44,-0.12256795355510364,778,True), Edge(36,62,-0.8530147925206462,1296,False), Edge(34,45,0.5794853784089509,797,False), Edge(68,43,0.3059263268963195,807,False), Edge(0,43,-0.6609348240912036,808,False), Edge(57,66,0.2191665555241864,1320,False), Edge(58,44,0.11138842776042979,1319,False), Edge(68,45,-0.8765888714500634,812,False), Edge(55,43,-0.9995351101645054,813,False), Edge(28,43,-0.5475180495352123,302,True), Edge(69,43,-0.17414989748242515,814,True), Edge(62,45,0.44380484453692914,304,False), Edge(2,43,-0.09266398718952162,817,True), Edge(76,43,0.7983282107073246,1326,False), Edge(13,44,0.31265304875226807,307,False), Edge(70,43,0.6678264940348166,821,True), Edge(62,44,-0.7068018849469007,312,False), Edge(49,44,0.5359946794582107,313,True), Edge(26,45,0.869472196793474,314,False), Edge(45,44,-0.6032023549759922,315,False), Edge(25,43,0.8003245490989297,318,True), Edge(8,62,0.331925224898749,320,False), Edge(63,44,-0.2836291462564269,321,False), Edge(25,44,0.31348075752476046,322,True), Edge(57,44,0.6604283431806024,834,False), Edge(19,45,-0.10450404928162582,836,False), Edge(50,44,0.5359946794582107,325,True), Edge(57,45,0.5999593672962404,838,True), Edge(70,45,0.4701113496165852,832,False), Edge(72,66,0.8513855240349133,1347,False), Edge(46,66,-0.01557537255413366,327,False), Edge(79,43,-0.3650279315931255,328,False), Edge(71,43,0.476783050699243,845,False), Edge(45,43,-0.9367781546143004,334,True), Edge(53,68,0.3112582897720597,1361,False), Edge(81,43,-0.5322439970672019,337,False), Edge(13,45,-0.14706231130997804,339,False), Edge(36,45,-0.019019262905032974,342,True), Edge(28,44,-0.06784561366613251,343,True), Edge(76,43,-0.3650279315931255,1368,False), Edge(79,45,0.2563919412470079,351,False), Edge(64,43,0.4346767335815056,355,False), Edge(63,43,-0.5322439970672019,358,True), Edge(42,44,0.33453828833952937,361,True), Edge(11,66,-0.20719569292392825,362,False), Edge(27,43,0.11041161887496531,364,True), Edge(63,44,0.681234411269932,365,True), Edge(82,43,-0.3982579307394478,370,False), Edge(54,62,0.5355374920539491,377,False), Edge(49,64,0.9521978704291929,1403,False), Edge(64,44,-0.9308612057500862,382,False), Edge(7,43,-0.7097353472323571,383,True), Edge(54,43,0.6678264940348166,896,False), Edge(70,45,0.2563919412470079,894,True), Edge(49,43,0.9119771098216856,388,False), Edge(76,43,0.4628366522332579,1414,False), Edge(38,45,0.3806436348536191,903,False), Edge(46,43,0.7685685187134161,394,True), Edge(32,43,0.39268373828408487,396,True), Edge(45,43,-0.07759295731774696,399,True), Edge(46,43,-0.8077736831704678,404,True), Edge(47,43,-0.3982579307394478,406,False), Edge(38,68,0.8662768848412743,922,False), Edge(77,44,-0.4672960021894921,1436,False), Edge(22,45,-0.5676178283239404,413,True), Edge(65,64,0.4363506447264971,416,False), Edge(78,64,0.9985712391172805,1440,False), Edge(60,62,-0.9058796197218013,418,False), Edge(13,43,-0.3150032399572693,420,True), Edge(58,43,0.3786100821915166,935,True), Edge(39,43,-0.5666545360906394,939,False), Edge(60,44,-0.04985801851320115,428,False), Edge(71,44,-0.2293826077164396,940,False), Edge(80,45,0.2563919412470079,430,False), Edge(51,43,0.04697766191345032,433,True), Edge(8,45,0.7440236123580743,435,True), Edge(50,44,0.007545548730783613,436,True), Edge(19,45,1,953,True), Edge(48,43,-0.053043942192943616,447,True), Edge(52,44,-0.992296761553189,448,True), Edge(53,64,0.5243855844088463,960,False), Edge(59,43,0.13035685986078116,967,False), Edge(0,45,0.3645069602316928,461,True), Edge(52,43,0.14349992661864408,483,False), Edge(53,44,0.24083053459835768,492,True), Edge(34,44,0.373695706983159,498,False), Edge(72,64,0.373695706983159,1011,False), Edge(65,44,0.0008063182694602045,501,False), Edge(53,43,1,502,True), Edge(72,43,-0.7820140947565839,1017,False), Edge(12,44,-0.7482366356210437,511,True), Edge(83,66,0.3145928084083378,451,False), Edge(83,43,-0.5726435212213377,473,False), Edge(38,44,0.6683597794938083,569,False), Edge(55,66,0.5725332981088795,432,False), Edge(52,66,-0.6890681887942174,561,False), Edge(55,82,0.9792759687756791,464,False), Edge(20,66,0.3315531881952558,345,False), Edge(82,45,-0.8339587487547384,442,False)]
    nn.init(*edges)
    theta = playerPos

    def makeNewLine(boxes,nowRound,seed=SEED):
        boxes.pop()
        for y in boxes:
            for box in y:
                if type(box) == Box:
                    box.skinUpdate(nowRound)
        if sum(map(lambda x : type(x) == Box,boxes[-1])):
            return GAME_OVER
        newLine = [None]*WIDTH
        lineNum = set(range(WIDTH))
        random.seed(seed+nowRound)
        seed = [random.random() for i in range(EXPAND)][0]
        random.seed(seed+nowRound)
        idx = random.randint(0,WIDTH-1)
        newLine[idx] = Item()
        lineNum.remove(idx)
        lineNum = list(lineNum)
        random.seed(seed+nowRound)
        random.shuffle(lineNum)
        random.seed(seed+nowRound)
        for i in lineNum[:random.randint(1,WIDTH-1)]:
            newLine[i] = Box(nowRound)
        boxes.insert(0,newLine)
    
    def drawBox(screen,boxes):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if boxes[y][x] is not None:
                    boxes[y][x].draw(screen,(x,y))
    makeNewLine(boxes,1)
    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        mouse, eventKeys, pressed_keys = keyboard()
        screen.fill(WHITE)
        Bullet(moveToPos,math.radians(90)).draw(screen)
        #forwardResult = forward(boxes,playerPos,ballNumber,nowRound,nn)
        forwardResult = forward(boxes,playerPos,nn)
        showTheta = max(min(math.pi-forwardResult[0]*math.pi-math.pi/2,math.pi*(180-DEGREE_LIMIT)/180),math.pi*DEGREE_LIMIT/180)
        pygame.draw.line(screen,BLUE,(int(moveToPos),int(startPos)),(int(moveToPos) - math.cos(showTheta)*50, int(startPos) - math.sin(showTheta)*50))
        if (not nowShooting):
            theta = min(max(-forwardResult[0]*math.pi-math.pi/2,-math.pi*(180-DEGREE_LIMIT)/180),-math.pi*DEGREE_LIMIT/180)
            print(-theta*180/math.pi,forwardResult[0]*180+90)
            bullets.append(Bullet(playerPos,theta))
            delay = 10
            nowShooting = True
            cnt += 1
        elif nowShooting:
            if (not delay):
                if cnt < ballNumber:
                    bullets.append(Bullet(playerPos,theta))
                    delay = 10
                    cnt += 1
            else:
                delay -= 1
        popped = 0
        for i in range(len(bullets)):
            idx = i - popped
            bullet = bullets[idx]
            move = bullet.move(boxes)
            if move is not None:
                box = boxes[move[0]][move[1]]
                box.hp -= 1
                if type(box) == Box:
                    box.skinUpdate(nowRound)
                if box.hp <= 0 :
                    if type(box) == Item:
                        addBall += 1
                    boxes[move[0]][move[1]] = None
            if bullet.y >= startPos:
                if not comeBackBall : moveToPos = bullet.x
                bullets.pop(idx)
                comeBackBall += 1
                if comeBackBall == cnt:
                    cnt = 0
                    playerPos = moveToPos
                    comeBackBall = 0
                    nowShooting = False
                    ballNumber += addBall
                    addBall = 0
                    nowRound += 1
                    if makeNewLine(boxes,nowRound) == GAME_OVER:
                        print("YOU DIE")
                        sys.exit()
                    if sum(map(lambda x : type(x) == Item,boxes[-1])):
                        ballNumber += 1
                        boxes[-1] = [None]*WIDTH
                popped += 1
            bullet.draw(screen)
        drawBox(screen,boxes)
        pygame.display.update()