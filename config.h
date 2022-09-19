#pragma once

#define tileWidth		24
#define tileHeight		24

// windowR: used for dark image generation.
#define windowR			4
#define windowWidth		(1 + windowR * 2)
#define windowHeight	(1 + windowR * 2)
#define windowSize		(windowWidth * windowHeight)
#define aproneWidth		(tileWidth + windowR * 2)  // not larger than 78, shared memory limitation.
#define aproneHeight	(tileHeight + windowR * 2)
#define aproneSize		(aproneWidth * aproneHeight)

// guided_windowR: used for guided image filtering.
#define guided_windowR			windowR * 4
#define guided_windowWidth		(1 + guided_windowR * 2)
#define guided_windowHeight		(1 + guided_windowR * 2)
#define guided_windowSize		(guided_windowWidth * guided_windowHeight)
#define guided_aproneWidth		(tileWidth + guided_windowR * 2)  // not larger than 78, shared memory limitation.
#define guided_aproneHeight		(tileHeight + guided_windowR * 2)
#define guided_aproneSize		(guided_aproneWidth * guided_aproneHeight)

// pixel hitogram.
#define nbins  256

// parameters.
constexpr float ttilde_w = 0.95;
constexpr float guided_epsilon = 0.000001;
constexpr float J_t0 = 0.1;


