#pragma once

#define tileWidth		23
#define tileHeight		23
#define windowR			4
#define windowWidth		(1 + windowR * 2)
#define windowHeight	(1 + windowR * 2)
#define windowSize		(windowWidth * windowHeight)
#define aproneWidth		(tileWidth + windowR * 2)  // not larger than 78, shared memory limitation.
#define aproneHeight	(tileHeight + windowR * 2)

