import pygame,sys

def write(screen,text,pos,font,size,color,center = None):
    try:
        myfont = pygame.font.Font(font, size)
    except:
        myfont = pygame.font.SysFont(font, size)

    textsurface = myfont.render(text, False, color)
    if center is None:
        text_rect = textsurface.get_rect(center=(pos[0], pos[1]))
        screen.blit(textsurface,text_rect)
    elif center:
        screen.blit(textsurface,pos)
    else:
        text_rect = textsurface.get_rect()
        text_rect.right = pos[0]
        screen.blit(textsurface,text_rect)

def keyboard():
    mouse = [0, 0, 0]
    eventKeys = []
    pressed_keys = pygame.key.get_pressed()
    mouse = list(pygame.mouse.get_pos()) + [pygame.mouse.get_pressed()[0]]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            eventKeys.append(event.key)
    return mouse, eventKeys, pressed_keys

class Timer:
    def __init__(self,setting) -> None:
        self.t = 0
        self.setting = setting
        