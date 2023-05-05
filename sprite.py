from parameters import *
from system import *
import math

def ballHit(bx,by,size,x,y,radius):
    r = radius
    x1 = bx - size[0]/2
    x2 = bx + size[0]/2
    y1 = by + size[1]/2
    y2 = by - size[1]/2
    ex1 = x1 - r
    ex2 = x2 + r
    ey1 = y1 + r
    ey2 = y2 - r
    if (ex1 <= x <= x1 and y2 <= y <= y1):
        return 1
    if (x2 <= x <= ex2 and y2 <= y <= y1):
        return 1
    if (x1 <= x <= x2 and ey2 <= y <= y2):
        return 2
    if (x1 <= x <= x2 and y1 <= y <= ey1):
        return 2
    if ((x - x1)**2 + (y - y1)**2 <= r**2) or ((x - x2)**2 + (y - y1)**2 <= r**2) or ((x - x1)**2 + (y - y2)**2 <= r**2) or ((x - x2)**2 + (y - y2)**2 <= r**2):
        return 3
    if (x1 <= x <= x2 and y1 <= y <= y2):
        return 4
    return False

class Sprite:
    def __init__(self,x,y,xSize,ySize) -> None:
        self.x = x
        self.y = y
        self.size = (xSize,ySize)
        self.color = WHITE
        self.skin = pygame.Surface((xSize,ySize),pygame.SRCALPHA)
    
    def draw(self,screen):
        self.skin.set_alpha(255)
        screen.blit(self.skin,(self.x-self.size[0]/2,self.y-self.size[1]/2))

class Platform(Sprite):
    def __init__(self, x, y, xSize, ySize,color=BLACK) -> None:
        super().__init__(x, y, xSize, ySize)
        self.color = color
        pygame.draw.rect(self.skin,self.color,pygame.Rect(0,0,xSize,ySize))

class Bullet(Sprite):
    def __init__(self, x, direction, speed = 10) -> None:
        super().__init__(x, SCREEN_HEIGHT - BULLET_SIZE/2, BULLET_SIZE, BULLET_SIZE)
        pygame.draw.circle(self.skin,BLUE,(BULLET_SIZE/2,BULLET_SIZE/2),BULLET_SIZE/2)
        self.radius = BULLET_SIZE/2
        self.direction = direction
        self.speed = speed
        self.isHitBox = False
    
    def move(self,boxes):
        dx = math.cos(self.direction) * self.speed
        dy = math.sin(self.direction) * self.speed
        move = [False,False]
        if abs(self.x + dx) < self.radius :
            self.x = - dx - self.x + self.radius
            move[0] = True
            self.direction = math.pi - self.direction
        if abs(self.x + dx - SCREEN_WIDTH) < self.radius :
            self.x += dx - (SCREEN_WIDTH - self.x + self.radius)
            move[0] = True
            self.direction = math.pi - self.direction
        c = BOX_SIZE//2
        if abs(self.y + dy - c) < self.radius :
            self.y += dy - (c - self.y + self.radius)
            move[1] = True
            self.direction = - self.direction
        for y in range(HEIGHT):
            for x in range(WIDTH):
                box = boxes[y][x]
                if type(box) == Box:
                    pos = (x*3*BOX_SIZE//2+box.size[0]/2,3*BOX_SIZE//2 + y*BOX_SIZE+box.size[1]/2)
                    rst = ballHit(pos[0], pos[1], box.size, self.x, self.y, self.radius)
                    if not self.isHitBox:
                        match rst:
                            case 1:
                                self.direction = math.pi - self.direction
                            case 2:
                                self.direction = - self.direction
                            case 3:
                                self.direction = math.pi - self.direction
                            case 4:
                                self.direction += math.pi
                        if rst:
                            dx = math.cos(self.direction) * self.speed
                            dy = math.sin(self.direction) * self.speed
                            self.x += dx * (not move[0])
                            self.y += dy * (not move[1])
                            return (y,x)
                    elif not rst:
                        self.isHitBox = False
                elif type(box) == Item:
                    pos = (x*3*BOX_SIZE//2 + BOX_SIZE/2,3*BOX_SIZE//2 + y*BOX_SIZE+BOX_SIZE/4)
                    if (pos[0] - self.x)**2 + (pos[1] - self.y)**2 <= (self.radius+BOX_SIZE*3/4)**2:
                        return (y,x)
        
        dx = math.cos(self.direction) * self.speed
        dy = math.sin(self.direction) * self.speed
        self.x += dx * (not move[0])
        self.y += dy * (not move[1])
        self.x = min(max(self.x,self.radius),SCREEN_WIDTH-self.radius)
        self.y = max(self.y,c)
        return None

class Item(Sprite):
    def __init__(self) -> None:
        super().__init__(0,0,BOX_SIZE/2, BOX_SIZE/2)
        self.color = GREEN
        self.hp = 1
        pygame.draw.circle(self.skin,self.color,(BOX_SIZE/4,BOX_SIZE/4),BOX_SIZE/4)

    def draw(self,screen,pos):
        self.skin.set_alpha(255)
        screen.blit(self.skin,(pos[0]*3*BOX_SIZE//2 + BOX_SIZE/2,3*BOX_SIZE//2 + pos[1]*BOX_SIZE+BOX_SIZE/4))

class Box(Sprite):
    def __init__(self, hp, xSize = BOX_SIZE*1.5, ySize = BOX_SIZE) -> None:
        super().__init__(0,0,xSize, ySize)
        self.hp = hp
        try:
            self.skinUpdate(hp)
        except:
            pass
    
    def skinUpdate(self,nowRound):
        self.skin = pygame.Surface(self.size,pygame.SRCALPHA)
        self.color = (255,127*(1-self.hp/nowRound),0)
        pygame.draw.rect(self.skin,self.color,pygame.Rect(0,0,self.size[0],self.size[1]))
        write(self.skin,str(self.hp),(self.size[0]//2,self.size[1]//2),'Consolas',self.size[1]//2,WHITE)

    def draw(self,screen,pos):
        self.skin.set_alpha(255)
        screen.blit(self.skin,(pos[0]*3*BOX_SIZE//2,3*BOX_SIZE//2 + pos[1]*BOX_SIZE))
