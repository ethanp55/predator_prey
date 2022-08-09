class Utils(object):
    PREY_NAME = 'prey'
    AVAILABLE = -1  # A cell in the grid with this value means it is available/not taken
    VERTICAL, HORIZONTAL = 'vertical', 'horizontal'
    POSSIBLE_MOVEMENTS = [VERTICAL, HORIZONTAL]  # Up/down, left/right
    POSSIBLE_DELTA_VALS = [-1, 0, 1]  # Left/down, stay, right/up
    MAX_MOVEMENT_UNITS = 1
