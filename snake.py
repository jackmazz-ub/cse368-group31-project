from enum import IntEnum

class Directions(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

# map directions to row/column distance
direc_dists = {
    Directions.NORTH: -1
    Directions.SOUTH: 1
    Directions.EAST: 1
    Directions.WEST: -1
}

# a single peice of the snake
# links to other segments like a linked list
class Segment:
    def __init__(self, row, col, direc):
        self.row = row
        self.col = col
        self.direc = direc
        self.link = None # subsequent segment
        
    # move this segment in a particular direction
    # move all subsequent segments in direction this was previously moving in
    def move(self, direc):
        link_i = self
        direc_i = direc
        
        # move each segment in the chain
        while True:
        
            # choose the next position
            row = direc_dists(link_i.row)
            col = direc_dists(link_i.col)
            
            # swap direc_i and link_i.direc
            direc_swap = link_i.direc
            link_i.direc = direc_i
            direc_i = direc_swap
            
            link_i = link_i.link
            if link_i is None:
                break
    
    # add segments to the tail of a chain
    def grow(self, length):
    
        # segments can only be added to the tail
        tail = self
        while tail.link is not None:
            tail = tail.link
        
        for i in range(length):
        
            # choose the initial position
            row = -1*direc_dists(link_i.row)
            col = -1*direc_dists(link_i.col)
            
            tail.link = Segment(row, col, self.direc)
            tail = tail.link

# a wrapper which abstracts the segment system
# should be used instead the segment class
class Snake:
    def __init__(self, length, row, col, direc):
        self.head = Segment(row, col, direc)
        self.tail = self.head
        self.tail.grow(length-1)
        self.length = length
    
    def move(self, direc):
        self.head.move(direc)
    
    def grow(self, length):
        self.tail.grow(length)
        self.length += length
    
    def __len__(self):
        return self.length

    def __iter__(self):
        segment = self.head
        while segment is not None:
            yield segment
            segment = segment.link
    
