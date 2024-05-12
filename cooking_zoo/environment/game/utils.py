import pygame
from cooking_zoo.cooking_world.actions import *


class Color:
    BLACK = (0, 0, 0)
    FLOOR = (245, 230, 210)  # light gray
    COUNTER = (220, 170, 110)  # tan/gray
    COUNTER_BORDER = (114, 93, 51)  # darker tan
    DELIVERY = (96, 96, 96)  # grey


KeyToTuple = {
    pygame.K_UP: (0, -1),  # 273
    pygame.K_DOWN: (0, 1),  # 274
    pygame.K_RIGHT: (1, 0),  # 275
    pygame.K_LEFT: (-1, 0),  # 276
}

KeyToTuple_human1 = {
    pygame.K_UP: ActionScheme1.WALK_UP,  # 273
    pygame.K_DOWN: ActionScheme1.WALK_DOWN,  # 274
    pygame.K_RIGHT: ActionScheme1.WALK_RIGHT,  # 275
    pygame.K_LEFT: ActionScheme1.WALK_LEFT,  # 276
    pygame.K_SPACE: ActionScheme1.NO_OP,
    pygame.K_f: ActionScheme1.INTERACT_PRIMARY,
    pygame.K_g: ActionScheme1.INTERACT_PICK_UP_SPECIAL,
    pygame.K_e: ActionScheme1.EXECUTE_ACTION
}

KeyToTuple_Scheme2_human1 = {
    pygame.K_UP: ActionScheme2.WALK_UP,  # 273
    pygame.K_DOWN: ActionScheme2.WALK_DOWN,  # 274
    pygame.K_RIGHT: ActionScheme2.WALK_RIGHT,  # 275
    pygame.K_LEFT: ActionScheme2.WALK_LEFT,  # 276
    pygame.K_SPACE: ActionScheme2.NO_OP,
    pygame.K_f: ActionScheme2.INTERACT_PRIMARY,
    pygame.K_g: ActionScheme2.INTERACT_PICK_UP_SPECIAL,
    pygame.K_e: ActionScheme2.EXECUTE_ACTION,
    pygame.K_o: ActionScheme2.COMMUNICATE_ZERO,
    pygame.K_i: ActionScheme2.COMMUNICATE_ONE
}

KeyToTuple_Scheme3_human1 = {
    pygame.K_UP: ActionScheme3.WALK_UP,  # 273
    pygame.K_DOWN: ActionScheme3.WALK_DOWN,  # 274
    pygame.K_RIGHT: ActionScheme3.WALK_RIGHT,  # 275
    pygame.K_LEFT: ActionScheme3.WALK_LEFT,  # 276
    pygame.K_SPACE: ActionScheme3.NO_OP,
    pygame.K_f: ActionScheme3.INTERACT_PRIMARY,
    pygame.K_g: ActionScheme3.INTERACT_PICK_UP_SPECIAL,
    pygame.K_h: ActionScheme3.EXECUTE_ACTION,
    pygame.K_q: ActionScheme3.COMMUNICATE_ZERO_0,
    pygame.K_w: ActionScheme3.COMMUNICATE_ONE_0,
    pygame.K_e: ActionScheme3.COMMUNICATE_ZERO_1,
    pygame.K_r: ActionScheme3.COMMUNICATE_ONE_1
}
KeyToTuple_human2 = {
    pygame.K_w: (0, -1),
    pygame.K_s: (0, 1),
    pygame.K_d: (1, 0),
    pygame.K_a: (-1, 0),
}
