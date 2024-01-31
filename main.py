import pygame
from abc import abstractmethod, ABC
from nn import Matrix, NeuralNetwork
from random import randint, random
from object import GameObject
from math import floor

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = 'Human VS AI'
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

AI_PILLAR_GAP = 150
PLAYER_PILLAR_GAP = 100
PILLAR_WIDTH = 50
NO_GAP_MARGIN = 50
PILLAR_SPEED = 1
PILLAR_COUNT = 4
PILLAR_INTERVAL = WINDOW_WIDTH / PILLAR_COUNT

BIRD_SIZE = 30
BIRD_X_POSITION = 100

GRAVITY = 10
ACCELERATION = 0.2
JUMP_HEIGHT = 4

SEP_LINE_HEIGHT = 5

ENEMY_BIRDS = 100

BIRD_BR_INPUTS = 5
BIRD_BRAIN_CFG = [
    {
        'nodes': 10,
        'activation_func': 'sigmoid'
    },
    {
        'nodes': 10,
        'activation_func': 'sigmoid'
    },
    {
        'nodes': 2,
        'activation_func': 'sigmoid'
    }
]

class Window(ABC):
    def __init__(self, width, height, title, fps):
        self.width = width
        self.height = height
        self.title = title

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = fps

    def set_fps(self, fps):
        self.fps = fps

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def click(self, mouse_position):
        pass

    @abstractmethod
    def key_pressed(key):
        pass

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_position = pygame.mouse.get_pos()
                    self.click(mouse_position)
                if event.type == pygame.KEYDOWN:
                    self.key_pressed(event.key)
            self.update()
            self.draw()

            pygame.display.flip()

            self.clock.tick(self.fps)

        pygame.quit()

    def text(self, txt, size, x, y, color):
        font = pygame.font.Font('freesansbold.ttf', size)
        t = font.render(txt, True, color)
        tr = t.get_rect()
        tr.center = (x, y)
        self.screen.blit(t, tr)

class Game(ABC):
    def __init__(self, y, height, pillar_gap, gc):
        self.y = y
        self.height = height
        self.gc = gc
        self.pillars = []
        self.score = 0
        self.best_score = 0
        self.pillar_gap = pillar_gap
        for i in range(PILLAR_COUNT):
            self.spawn_pillar()

    def reset(self):
        self.pillars = []
        for i in range(PILLAR_COUNT):
            self.spawn_pillar()
        self.score = 0

    def spawn_pillar(self):
        x = self.gc.width / 2
        if len(self.pillars) > 0:
            last_index = len(self.pillars) - 1
            x = self.pillars[last_index].upper_part.x + PILLAR_INTERVAL
        self.pillars.append(Pillar(x, self))

    def draw(self):
        for pillar in self.pillars:
            pillar.draw()
        self.gc.text(
            f'Score: {self.score}   Best Score: {self.best_score}', 
            10,
            self.gc.width / 2,
            self.y + self.height / 2, 
            BLACK
        )

    def update(self):
        for pillar in self.pillars:
            pillar.update()
        if self.score >= self.best_score:
            self.best_score = self.score

    def spawn_bird(self):
        return Bird(BIRD_X_POSITION, self.y +  self.height / 2, self)
    
    def spawn_ai_bird(self):
        return BirdAI(BIRD_X_POSITION, self.y + self.height / 2, self)
    
    def get_closest_pillar(self, bird):
        for pillar in self.pillars:
            if pillar.get_x() > bird.x:
                return pillar


class PillarPart(GameObject):
    def __init__(self, x, y, height, game):
        super().__init__(x, y, PILLAR_WIDTH, height, game)

    def draw(self):
        rect = (self.x, self.y, self.width, self.height)
        pygame.draw.rect(self.gc.screen, BLACK, rect)

    def update(self):
        self.move(-PILLAR_SPEED, 0)


class Pillar:
    def __init__(self, x, game):
        self.game = game
        self.generate_parts(x)

    def generate_parts(self, x):
        start = NO_GAP_MARGIN
        end = self.game.height - self.game.pillar_gap - NO_GAP_MARGIN
        upper_part_height = randint(start, end)
        lower_part_height = self.game.height - upper_part_height - self.game.pillar_gap
        
        y = self.game.y
        height = upper_part_height

        self.upper_part = PillarPart(x, y, height, self.game)

        y = self.game.y + upper_part_height + self.game.pillar_gap
        height = lower_part_height

        self.lower_part = PillarPart(x, y, height, self.game)
        
    def draw(self):
        self.upper_part.draw()
        self.lower_part.draw()

    def update(self):
        self.upper_part.update()
        self.lower_part.update()
        if self.upper_part.x < -PILLAR_WIDTH:
            self.generate_parts(self.game.gc.width - PILLAR_WIDTH)

    def get_x(self):
        return self.upper_part.x
    
    def get_low_upper_pillar_y(self):
        return self.upper_part.y + self.upper_part.height


class Population:
    def __init__(self, units, mutation_rate, game):
        self.birds = []
        self.units = units
        for i in range(units):
            self.birds.append(game.spawn_ai_bird())
        self.mutation_rate = mutation_rate
        self.best = self.birds[0]
        self.game = game

    def next_generation(self):
        mating_pool = []
        max_fitness = 1
        for bird in self.birds:
            if bird.fitness > max_fitness:
                max_fitness = bird.fitness

        for bird in self.birds:
            fitness = bird.fitness / max_fitness
            n = floor(fitness * 100)
            for i in range(n) :
                mating_pool.append(bird)

        for i in range(len(self.birds)):
            a = randint(0, len(mating_pool) - 1)
            b = randint(0, len(mating_pool) - 1)

            parent_a = mating_pool[a]
            parent_b = mating_pool[b]

            child = self.crossover(parent_a, parent_b)
            if random() < self.mutation_rate:
                child.mutate()

            self.birds[i] = child

    def find_best(self):
        self.best = self.birds[0]
        for bird in self.birds[1:]:
            if bird.fitness > self.best.fitness:
                self.best = bird

    def crossover(self, a, b):
        child = self.game.spawn_ai_bird()
        midpoint = randint(0, len(a.brain.layers))

        for i in range(len(child.brain.layers)):
            if i < midpoint:
                child.brain.layers[i].weights = a.brain.layers[i].weights.copy()
                child.brain.layers[i].bias = a.brain.layers[i].bias.copy()
            else:
                child.brain.layers[i].weights = b.brain.layers[i].weights.copy()
                child.brain.layers[i].bias = b.brain.layers[i].bias.copy()

        return child

    def birds_alive(self):
        alive_count = sum(1 for bird in self.birds if not bird.dead)
        return alive_count
    

class Bird(GameObject):
    def __init__(self, x, y, game):
        super().__init__(x, y, BIRD_SIZE, BIRD_SIZE, game)
        self.dead = False
        self.y_velocity = 0

    def draw(self):
        if self.dead:
            return
        x = self.x + BIRD_SIZE / 2
        y = self.y + BIRD_SIZE / 2
        pygame.draw.circle(self.gc.screen, BLACK, (x, y), BIRD_SIZE / 2)

    def update(self):
        if self.dead:
            return
        if self.y_velocity < GRAVITY:
            self.y_velocity += ACCELERATION
        self.y += self.y_velocity
        for pillar in self.game.pillars:
            if self.collides(pillar.upper_part) or self.collides(pillar.lower_part):
                self.dead = True
        if self.y < self.game.y or self.y > self.game.y + self.game.height:
            self.dead = True

    def jump(self):
        self.y_velocity = -JUMP_HEIGHT

class BirdAI(Bird):
    def __init__(self, x, y, game):
        super().__init__(x, y, game)
        self.brain = NeuralNetwork(BIRD_BRAIN_CFG, 
            input_nodes=BIRD_BR_INPUTS
        )
        self.fitness = 0

    def update(self):
        super().update()
        if not self.dead:
            self.fitness += PILLAR_SPEED

    def think(self):
        closest_pillar = self.game.get_closest_pillar(self)
        prediction = self.brain.predict([
            closest_pillar.get_x(),
            closest_pillar.get_low_upper_pillar_y(),
            closest_pillar.lower_part.y,
            closest_pillar.get_x() - self.x,
            self.y
        ])

        if(prediction[0] > prediction[1]):
            self.jump()

    def mutate(self):
        for layer in self.brain.layers:
            layer.weights.random(-1, 1)
            layer.bias.random(-1, 1)


class AIGame(Game):
    def __init__(self, gc):
        super().__init__(0, gc.height / 2, AI_PILLAR_GAP, gc)
        self.population = Population(ENEMY_BIRDS, 0.005, self)

    def draw(self):
        super().draw()
        for bird in self.population.birds:
            bird.draw()

    def update(self):
        super().update()
        for bird in self.population.birds:
            bird.think()
            bird.update()
        self.population.find_best()
        self.score = self.population.best.fitness
        if self.population.birds_alive() == 0:
            self.reset()
            self.population.next_generation()


class PlayerGame(Game):
    def __init__(self, gc):
        super().__init__(gc.height / 2, gc.height / 2, PLAYER_PILLAR_GAP, gc)
        self.bird = self.spawn_bird()

    def draw(self):
        super().draw()
        self.bird.draw()

    def update(self):
        super().update()
        self.bird.update()
        self.score += PILLAR_SPEED
        if self.bird.dead:
            self.reset()
            self.bird = self.spawn_bird()


class GameController(Window):
    def __init__(self):
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, FPS)
        self.ai_game = AIGame(self)
        self.player_game = PlayerGame(self)
        self.started = False
        self.paused = False

    def click(self, mouse_position):
        pass

    def sep_line(self):
        y = self.height / 2 - SEP_LINE_HEIGHT / 2
        line = pygame.Rect(0, y, self.width, SEP_LINE_HEIGHT)
        return line

    def draw(self):
        self.screen.fill(WHITE)
        if not self.started:
            self.text('Press SPACE to start.', 30, self.width / 2, self.height / 2, BLACK)
        else:
            self.ai_game.draw()
            self.player_game.draw()
            pygame.draw.rect(self.screen, BLACK, self.sep_line())

    def key_pressed(self, key):
        if key == pygame.K_SPACE:
            if self.started:
                self.player_game.bird.jump()
            else:
                self.started = True
        if key == pygame.K_r and self.started:
            self.started = False
            self.paused = False
            self.player_game = PlayerGame(self)
            self.ai_game = AIGame(self)
        if key == pygame.K_p and self.started:
            self.paused = not self.paused

    def update(self):
        if self.started:
            self.ai_game.update()
            if not self.paused:
                self.player_game.update()

if __name__ == '__main__':
    pygame.init()
    GameController().run()
    