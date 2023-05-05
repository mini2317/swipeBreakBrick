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
    edges = [Edge(18,43,1,0,True), Edge(34,45,0.21058500127923674,1025,False), Edge(56,47,0.6570076750245359,1028,True), Edge(64,47,0.39069631814296235,1031,False), Edge(47,43,0.5003362333044425,520,True), Edge(5,43,-0.8371374350749323,522,False), Edge(19,44,0.6565178264768332,1037,False), Edge(5,44,-0.4705331555804675,526,True), Edge(40,43,-0.4765639163473838,1045,True), Edge(59,47,0.26740726951080696,538,True), Edge(28,63,-0.4123065029117914,1053,False), Edge(49,44,-0.42540059482972525,541,True), Edge(62,43,-0.8119777408758064,1054,True), Edge(32,44,-0.965653190871629,551,True), Edge(45,47,-0.04641306846399296,552,True), Edge(39,43,0.7596962626003181,1063,False), Edge(4,47,0.021212703451514603,554,True), Edge(44,43,-1,49,False), Edge(15,47,0.8822822032234725,568,True), Edge(55,47,-0.17737544828192453,1594,False), Edge(58,47,-0.041630651261621354,1089,False), Edge(38,45,0.5990178322706388,582,True), Edge(64,43,-0.2643162514300994,1096,False), Edge(44,43,-0.27397489898011496,73,True), Edge(15,45,0.9123396568031223,1109,True), Edge(65,47,0.21329960027319905,1112,False), Edge(44,43,-0.22378758071834914,97,True), Edge(50,44,-0.1235970374370623,612,True), Edge(59,44,-0.9909270340756879,615,True), Edge(65,48,-0.5388985606323471,1130,False), Edge(5,48,0.1407198100650433,1137,False), Edge(26,45,0.3850458010872666,631,True), Edge(48,47,-0.7060799401805973,633,True), Edge(14,44,-0.303352703550283,1157,False), Edge(57,44,-0.7878182557766358,1158,False), Edge(54,47,0.6473027463771746,649,True), Edge(16,47,-0.4247808367844943,1162,False), Edge(55,45,-0.9163435812607001,1166,False), Edge(51,44,0.8930297944132561,658,True), Edge(63,43,0.41706484833615454,1173,False), Edge(55,43,0.0971317118205548,665,True), Edge(57,48,-0.5734913284495591,666,False), Edge(44,43,-0.415950218967879,160,True), Edge(60,45,0.11820232406317577,680,False), Edge(66,44,-0.8132156056241093,1201,False), Edge(38,47,-0.7862273546436933,697,True), Edge(52,44,-0.02708780243655662,702,True), Edge(61,47,0.7443108410420107,706,True), Edge(64,45,-0.6545210347133859,1218,False), Edge(49,48,0.26909858598265557,710,False), Edge(18,47,-0.11147643908687144,712,True), Edge(67,44,0.7233720092986469,1244,False), Edge(58,48,0.625147790811621,750,True), Edge(53,43,-0.08853303316528494,762,False), Edge(6,47,0.5074916573739399,794,True), Edge(11,48,0.8439026729847507,833,True), Edge(61,43,0.8863691271672718,834,True), Edge(19,48,-0.7906145547910879,837,False), Edge(57,43,0.1982014313514615,330,False), Edge(54,43,0.13672538956302493,852,True), Edge(45,44,0.22893340373369808,350,True), Edge(1,43,0.44452606368743175,354,True), Edge(33,47,0.10226422467064444,1381,False), Edge(52,43,0.9723064875252698,884,True), Edge(22,45,0.785416270463597,388,True), Edge(18,44,0.017490893211997482,390,True), Edge(16,44,0.18984224833343877,906,False), Edge(45,43,0.22346407183743522,401,True), Edge(5,47,-0.24211562618634797,915,False), Edge(46,43,-0.409855192703517,412,True), Edge(57,43,0.03680788254996448,414,False), Edge(46,43,-0.6264131929218959,423,True), Edge(3,44,0.24567202110460795,427,True), Edge(53,48,0.10454601694767107,939,True), Edge(13,43,-0.6007149532665506,430,False), Edge(8,45,-0.6332908548554284,433,False), Edge(68,47,0.6473027463771746,1458,False), Edge(28,45,-0.023374089708153356,949,False), Edge(55,48,0.3128895187201173,951,True), Edge(57,47,0.26740726951080696,444,True), Edge(42,44,0.5627501817686282,451,False), Edge(6,43,-0.9070302637301442,452,False), Edge(28,48,0.3013660042939659,454,True), Edge(68,43,0.9010695495035401,1481,True), Edge(35,63,-0.8242170997559968,1483,False), Edge(58,43,0.8688593989536768,460,True), Edge(68,48,0.4534097630079479,1484,False), Edge(32,43,-0.46381371638647284,975,True), Edge(23,44,0.382304794883759,464,True), Edge(58,43,-1,474,True), Edge(62,47,0.4455953398476922,994,False), Edge(38,43,0.9010695495035401,483,True), Edge(69,45,0.785416270463597,1509,False), Edge(29,44,-0.5028323420909686,489,False), Edge(68,45,0.19434033854680854,1518,False), Edge(63,47,-0.23961542532272073,1007,False), Edge(47,44,0.7903263978226183,510,True), Edge(70,47,-0.12713758353604887,1679,False), Edge(70,43,-0.016592061459720808,1425,False), Edge(62,44,0.7233834226640583,1367,False), Edge(68,43,-0.40080351940636594,1368,False), Edge(63,44,0.7571193867416197,1625,False), Edge(72,44,0.6684859076810368,1629,False), Edge(62,43,-0.13557634849586564,1181,True), Edge(51,45,0.40285391462202114,1249,False), Edge(73,43,0.10218280052859119,1639,True), Edge(12,45,0.747542899470431,1258,False), Edge(74,43,-0.4128927711943946,1708,True), Edge(70,43,0.9010695495035401,1581,False), Edge(63,43,0.41706484833615454,1197,False), Edge(55,63,-0.09745110070119067,1583,False), Edge(30,45,0.16728020144018085,1132,False), Edge(66,72,0.49532263942204047,1517,False), Edge(65,43,0.9329617965078452,1263,False), Edge(69,43,-0.3444960333705793,1393,False), Edge(71,43,-0.4128927711943946,1460,True), Edge(35,73,-0.7146242258648232,1652,False), Edge(41,63,-0.7071290493945388,1656,False), Edge(30,45,-0.24661639634768573,1273,False), Edge(71,45,0.6176993965477315,1595,False), Edge(68,44,0.18984224833343877,1340,False), Edge(72,45,0.34068837493151505,1469,False), Edge(75,43,-0.4128927711943946,1726,False), Edge(23,73,-0.1582184429697766,1763,False), Edge(73,76,1.0,1771,False), Edge(76,43,0.10218280052859119,1771,False)]
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
        random.seed(seed)
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