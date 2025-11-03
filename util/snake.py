from enum import IntEnum

class Directions(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

# a single peice of the snake
# links to other segments like a linked list
class Segment:
    def __init__(self, row, col, direction):
        self.row = row
        self.col = col
        self.direction = direction
        self.link = None # subsequent segment
        
    # move this segment in a particular direction
    # move all subsequent segments in direction this was previously moving in
    def move(self, direction):   
        # choose the next position
        match direction:
            case Directions.NORTH:
                self.row = row+1
            case Directions.SOUTH:
                self.row = row-1
            case Directions.EAST:
                self.col = col+1
            case Directions.WEST:
                self.col = col-1
        
        # move subsequent segments
        link_i = self.link
        while link_i is not None:
            link_i.move(self.direction)
            link_i = link_i.link
        self.direction = direction
    
    # add segments to the tail of a chain
    def grow(self, length):
    
        # segments can only be added to the tail
        tail = self
        while tail.link is not None:
            tail = tail.link
        
        for i in range(length):
        
            # add the segment in the opposite direction which this is moving
            row = tail.row
            col = tail.col
            match self.direction:
                case Directions.NORTH:
                    row = row-1
                case Directions.SOUTH:
                    row = row+1
                case Directions.EAST:
                    col = col-1
                case Directions.WEST:
                    col = col+1
            tail.link = Segment(row, col, self.direction)
            tail = tail.link

# a wrapper which abstracts the segment system
# should be used instead the segment class
class Snake:
    def __init__(self, length, row, col, direction):
        self.head = Segment(row, col, direction)
        self.tail = self.head
        self.tail.grow(length-1)
        self.length = length
    
    def move(self, direction):
        self.head.move(direction)
    
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
    
