
class ActionScheme1:

    WALK_UP = 4
    WALK_DOWN = 3
    WALK_RIGHT = 2
    WALK_LEFT = 1

    NO_OP = 0

    INTERACT_PRIMARY = 5
    INTERACT_PICK_UP_SPECIAL = 6
    EXECUTE_ACTION = 7

    WALK_ACTIONS = [WALK_UP, WALK_DOWN, WALK_RIGHT, WALK_LEFT]
    INTERACT_ACTIONS = [INTERACT_PRIMARY, INTERACT_PICK_UP_SPECIAL, EXECUTE_ACTION]
    ACTIONS = [NO_OP, WALK_LEFT, WALK_RIGHT, WALK_DOWN, WALK_UP, INTERACT_PRIMARY, INTERACT_PICK_UP_SPECIAL, EXECUTE_ACTION]


class ActionScheme2:

    WALK_UP = 4
    WALK_DOWN = 3
    WALK_RIGHT = 2
    WALK_LEFT = 1

    NO_OP = 0

    INTERACT_PRIMARY = 5
    INTERACT_PICK_UP_SPECIAL = 6
    EXECUTE_ACTION = 7

    COMMUNICATE_ZERO = 8
    COMMUNICATE_ONE = 9

    WALK_ACTIONS = [WALK_UP, WALK_DOWN, WALK_RIGHT, WALK_LEFT]
    INTERACT_ACTIONS = [INTERACT_PRIMARY, INTERACT_PICK_UP_SPECIAL, EXECUTE_ACTION]
    COMMUNICATE_ACTIONS = [COMMUNICATE_ZERO, COMMUNICATE_ONE]
    ACTIONS = [NO_OP, WALK_LEFT, WALK_RIGHT, WALK_DOWN, WALK_UP, INTERACT_PRIMARY, INTERACT_PICK_UP_SPECIAL,
               EXECUTE_ACTION, COMMUNICATE_ZERO, COMMUNICATE_ONE]


class ActionScheme3:
    WALK_UP = 4
    WALK_DOWN = 3
    WALK_RIGHT = 2
    WALK_LEFT = 1

    NO_OP = 0

    INTERACT_PRIMARY = 5
    INTERACT_PICK_UP_SPECIAL = 6
    EXECUTE_ACTION = 7

    COMMUNICATE_ZERO_0 = 8
    COMMUNICATE_ONE_0 = 9
    COMMUNICATE_ZERO_1 = 10
    COMMUNICATE_ONE_1 = 11
    COMMUNICATE_ZERO_2 = 12
    COMMUNICATE_ONE_2 = 13
    COMMUNICATE_ZERO_3 = 14
    COMMUNICATE_ONE_3 = 15
    COMMUNICATE_ZERO_4 = 16
    COMMUNICATE_ONE_4 = 17

    WALK_ACTIONS = [WALK_UP, WALK_DOWN, WALK_RIGHT, WALK_LEFT]
    INTERACT_ACTIONS = [INTERACT_PRIMARY, INTERACT_PICK_UP_SPECIAL, EXECUTE_ACTION]
    COMMUNICATE_ACTIONS = [COMMUNICATE_ZERO_0, COMMUNICATE_ONE_0, COMMUNICATE_ZERO_1, COMMUNICATE_ONE_1,
                           COMMUNICATE_ZERO_2, COMMUNICATE_ONE_2, COMMUNICATE_ZERO_3, COMMUNICATE_ONE_3,
                           COMMUNICATE_ZERO_4, COMMUNICATE_ONE_4]
    ACTIONS = [NO_OP, WALK_LEFT, WALK_RIGHT, WALK_DOWN, WALK_UP, INTERACT_PRIMARY, INTERACT_PICK_UP_SPECIAL,
               EXECUTE_ACTION, COMMUNICATE_ZERO_0, COMMUNICATE_ONE_0, COMMUNICATE_ZERO_1, COMMUNICATE_ONE_1,
               COMMUNICATE_ZERO_2, COMMUNICATE_ONE_2, COMMUNICATE_ZERO_3, COMMUNICATE_ONE_3, COMMUNICATE_ZERO_4,
               COMMUNICATE_ONE_4]


class ActionScheme4:

    WALK_UP = 4
    WALK_DOWN = 3
    WALK_RIGHT = 2
    WALK_LEFT = 1

    NO_OP = 0

    def __init__(self, world):
        self.world = world


