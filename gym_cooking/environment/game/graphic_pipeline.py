import pygame
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.misc.game.utils import *
from collections import defaultdict, namedtuple

import numpy as np
from pathlib import Path
import os.path


COLORS = ['blue', 'magenta', 'yellow', 'green']


def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
        canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
        image = pygame.image.load(canonicalized_path)
        _image_library[path] = image
    return image


GraphicsProperties = namedtuple("GraphicsProperties", ["pixel_per_tile", "holding_scale", "container_scale",
                                                       "width_pixel", "height_pixel", "tile_size", "holding_size",
                                                       "container_size", "holding_container_size"])


class GraphicPipeline:

    PIXEL_PER_TILE = 80
    HOLDING_SCALE = 0.5
    CONTAINER_SCALE = 0.7

    def __init__(self, env, display=False):
        self.env = env
        self.display = display
        self.screen = None
        self.graphics_dir = 'misc/game/graphics'
        self.graphics_properties = GraphicsProperties(self.PIXEL_PER_TILE, self.HOLDING_SCALE, self.CONTAINER_SCALE,
                                                      self.PIXEL_PER_TILE * self.env.unwrapped.world.width,
                                                      self.PIXEL_PER_TILE * self.env.unwrapped.world.height,
                                                      (self.PIXEL_PER_TILE, self.PIXEL_PER_TILE),
                                                      (self.PIXEL_PER_TILE * self.HOLDING_SCALE,
                                                       self.PIXEL_PER_TILE * self.HOLDING_SCALE),
                                                      (self.PIXEL_PER_TILE * self.CONTAINER_SCALE,
                                                       self.PIXEL_PER_TILE * self.CONTAINER_SCALE),
                                                      (self.PIXEL_PER_TILE * self.CONTAINER_SCALE * self.HOLDING_SCALE,
                                                       self.PIXEL_PER_TILE * self.CONTAINER_SCALE * self.HOLDING_SCALE))

    def on_render(self):
        self.screen.fill(Color.FLOOR)

        self.draw_static_objects()

        self.draw_agents()

        self.draw_dynamic_objects()

        if self.display:
            pygame.display.flip()
            pygame.display.update()

    def draw_square(self):
        pass

    def draw_static_objects(self):
        objects = self.env.unwrapped.world.get_object_list()
        static_objects = [obj for obj in objects if isinstance(obj, StaticObject)]
        for static_object in static_objects:
            self.draw_static_object(static_object)

    def draw_static_object(self, static_object: StaticObject):
        sl = self.scaled_location(static_object.location)
        fill = pygame.Rect(sl[0], sl[1], self.graphics_properties.pixel_per_tile,
                           self.graphics_properties.pixel_per_tile)
        if isinstance(static_object, Counter):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
        elif isinstance(static_object, DeliverSquare):
            pygame.draw.rect(self.screen, Color.DELIVERY, fill)
            self.draw('delivery', self.graphics_properties.tile_size, sl)
        elif isinstance(static_object, CutBoard):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('cutboard', self.graphics_properties.tile_size, sl)
        elif isinstance(static_object, Blender):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('blender3', self.graphics_properties.tile_size, sl)
        # elif isinstance(static_object, Floor):
        #     pygame.draw.rect(self.screen, Color.FLOOR, fill)

    def draw_dynamic_objects(self):
        objects = self.env.unwrapped.world.get_object_list()
        dynamic_objects = [obj for obj in objects if isinstance(obj, DynamicObject)]
        dynamic_objects_grouped = defaultdict(list)
        for obj in dynamic_objects:
            dynamic_objects_grouped[obj.location].append(obj)
        for location, obj_list in dynamic_objects_grouped.items():
            if any([agent.location == location for agent in self.env.unwrapped.world.agents]):
                self.draw_dynamic_object_stack(obj_list, self.graphics_properties.holding_size,
                                               self.holding_location(location),
                                               self.graphics_properties.holding_container_size,
                                               self.holding_container_location(location))
            else:
                self.draw_dynamic_object_stack(obj_list, self.graphics_properties.tile_size,
                                               self.scaled_location(location),
                                               self.graphics_properties.container_size,
                                               self.container_location(location))

    def draw_dynamic_object_stack(self, dynamic_objects, base_size, base_location, holding_size, holding_location):
        highest_order_object = self.env.unwrapped.world.get_highest_order_object(dynamic_objects)
        if isinstance(highest_order_object, Container):
            self.draw('Plate', base_size, base_location)
            rest_stack = [obj for obj in dynamic_objects if obj != highest_order_object]
            if rest_stack:
                file_name = self.get_file_name(rest_stack)
                self.draw(file_name, holding_size, holding_location)
        else:
            file_name = self.get_file_name(dynamic_objects)
            self.draw(file_name, base_size, base_location)

    def draw_agents(self):
        for agent in self.env.unwrapped.world.agents:
            self.draw('agent-{}'.format(agent.color), self.graphics_properties.tile_size,
                      self.scaled_location(agent.location))

    def draw(self, path, size, location):
        image_path = f'{self.root_dir}/{self.graphics_dir}/{path}.png'
        image = pygame.transform.scale(get_image(image_path), size)
        self.screen.blit(image, location)

    def draw_agent_object(self, obj):
        # Holding shows up in bottom right corner.
        if obj is None: return
        if any([isinstance(c, Plate) for c in obj.contents]):
            self.draw('Plate', self.graphics_properties.holding_size, self.holding_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge('Plate')
                self.draw(obj.full_name, self.graphics_properties.holding_container_size,
                          self.holding_container_location(obj.location))
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.graphics_properties.holding_size, self.holding_location(obj.location))

    def draw_object(self, obj):
        if obj is None:
            return
        if any([isinstance(c, Plate) for c in obj.contents]):
            self.draw('Plate', self.graphics_properties.tile_size, self.scaled_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge('Plate')
                self.draw(obj.full_name, self.graphics_properties.container_size, self.container_location(obj.location))
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.graphics_properties.tile_size, self.scaled_location(obj.location))

    @staticmethod
    def get_file_name(dynamic_objects):
        order = [Lettuce, Onion, Tomato]
        name = []
        for order_type in order:
            for obj in dynamic_objects:
                if isinstance(obj, order_type):
                    name.append(obj.state.value + obj.name())
                    break
        return '-'.join(name)

    def scaled_location(self, loc):
        """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
        return tuple(self.graphics_properties.pixel_per_tile * np.asarray(loc))

    def holding_location(self, loc):
        """Return top-left corner of location where agent holding will be drawn (bottom right corner)
        given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.graphics_properties.pixel_per_tile *
                      (1 - self.HOLDING_SCALE)).astype(int))

    def container_location(self, loc):
        """Return top-left corner of location where contained (i.e. plated) object will be drawn,
        given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.graphics_properties.pixel_per_tile *
                      (1 - self.CONTAINER_SCALE) / 2).astype(int))
    
    def holding_container_location(self, loc):
        """Return top-left corner of location where contained, held object will be drawn
        given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        factor = (1 - self.HOLDING_SCALE) + (1 - self.CONTAINER_SCALE) / 2 * self.HOLDING_SCALE
        return tuple((np.asarray(scaled_loc) + self.graphics_properties.pixel_per_tile * factor).astype(int))
