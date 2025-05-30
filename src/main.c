#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <mpfr.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

#include "thread-pool.h"

#define WIDTH  800
#define HEIGHT 600
// #define WIDTH 1600
// #define HEIGHT 900
// #define WIDTH  1920
// #define HEIGHT 1080

#define PRECISION_BITS 1024
#define ESCAPE_RADIUS 1e6

static int max_iter = 64;

static SDL_Window *window;
static SDL_Renderer *renderer;
static SDL_Texture *texture;

static atomic_int g_generation;
static atomic_bool g_orbit_ready;

static double *g_orbit_re;
static double *g_orbit_im;
static atomic_int g_orbit_amount;

static uint32_t pixels[WIDTH * HEIGHT];
static pthread_mutex_t pixels_mutex;

static int64_t pixels_done[WIDTH * HEIGHT];
static pthread_mutex_t pixels_done_mutex;

static inline uint32_t
interpolate_color (uint32_t c1, uint32_t c2, double frac)
{
  uint8_t r1 = (c1 >> 16) & 0xFF;
  uint8_t g1 = (c1 >> 8) & 0xFF;
  uint8_t b1 = c1 & 0xFF;

  uint8_t r2 = (c2 >> 16) & 0xFF;
  uint8_t g2 = (c2 >> 8) & 0xFF;
  uint8_t b2 = c2 & 0xFF;

  uint8_t r = (1.0f - frac) * r1 + frac * r2;
  uint8_t g = (1.0f - frac) * g1 + frac * g2;
  uint8_t b = (1.0f - frac) * b1 + frac * b2;

  return 0xFF000000 | (r << 16) | (g << 8) | b;
}

struct orbit_work
{
  mpfr_t center_re;
  mpfr_t center_im;
  double scale;
  double *orbit_re;
  double *orbit_im;
  int generation;
};

void
render_compute_orbit (mpfr_t center_re, mpfr_t center_im, double *orbit_re,
                      double *orbit_im, int generation)
{
  const double escape_radius_sq = ESCAPE_RADIUS * ESCAPE_RADIUS;

  mpfr_t z_re, z_im, temp_re, temp_im, re_sqr, im_sqr, escape_radius;
  mpfr_inits2 (PRECISION_BITS, z_re, z_im, temp_re, temp_im, re_sqr, im_sqr,
               escape_radius, (mpfr_ptr)0);

  mpfr_set_d (z_re, 0.0, MPFR_RNDN);
  mpfr_set_d (z_im, 0.0, MPFR_RNDN);

  mpfr_set_d (escape_radius, escape_radius_sq * escape_radius_sq, MPFR_RNDN);

  int iter = 0;

  while (iter < max_iter)
    {
      if (generation != atomic_load (&g_generation))
        return;

      double z_x = mpfr_get_d (z_re, MPFR_RNDN);
      double z_y = mpfr_get_d (z_im, MPFR_RNDN);

      orbit_re[iter] = z_x;
      orbit_im[iter] = z_y;

      mpfr_mul (re_sqr, z_re, z_re, MPFR_RNDN);
      mpfr_mul (im_sqr, z_im, z_im, MPFR_RNDN);

      mpfr_sub (temp_re, re_sqr, im_sqr, MPFR_RNDN);

      mpfr_mul (temp_im, z_re, z_im, MPFR_RNDN);
      mpfr_mul_ui (temp_im, temp_im, 2, MPFR_RNDN);

      mpfr_add (z_re, temp_re, center_re, MPFR_RNDN);
      mpfr_add (z_im, temp_im, center_im, MPFR_RNDN);

      mpfr_mul (re_sqr, z_re, z_re, MPFR_RNDN);
      mpfr_mul (im_sqr, z_im, z_im, MPFR_RNDN);
      mpfr_add (temp_re, re_sqr, im_sqr, MPFR_RNDN);

      if (mpfr_greater_p (temp_re, escape_radius))
        break;

      iter++;
    }
}

void
render_compute_orbit_thread (void *argument)
{
  struct orbit_work *work = argument;

  render_compute_orbit (work->center_re, work->center_im, work->orbit_re,
                        work->orbit_im, work->generation);

  if (work->generation != atomic_load (&g_generation))
    goto clean;

  pthread_mutex_lock (&pixels_mutex);
  memcpy (g_orbit_re, work->orbit_re, max_iter * sizeof (double));
  memcpy (g_orbit_im, work->orbit_im, max_iter * sizeof (double));
  atomic_store (&g_orbit_amount, max_iter);
  atomic_store (&g_orbit_ready, 1);
  pthread_mutex_unlock (&pixels_mutex);

clean:
  free (work->orbit_re);
  free (work->orbit_im);
  mpfr_clears (work->center_re, work->center_im, (mpfr_ptr)0);
  free (work);
}

struct render_work
{
  int x;
  int y;
  int tile;
  int step;
  int samples;
  double scale;
  double *orbit_re;
  double *orbit_im;
  int orbit_amount;
  int generation;
};

void
render_test (void *argument)
{
  struct render_work *work = argument;

  /*
  static const uint32_t palette[] = {
    0xFF000000, 0xFF1A0A5E, 0xFF3D1F99, 0xFF5C44C3, 0xFF7C68E5,
    0xFF9AA1F1, 0xFFB7BCFA, 0xFFDFE5FF, 0xFFB1C1D9, 0xFF7D91BF,
    0xFF4C65A7, 0xFF1F3D88, 0xFF0A1A5E,
  };
  */

  /*static const uint32_t palette[] = {
    0xBCCAB3,
    0x94A98F,
    0x6E8B6C,
    0x516A52,
    0x3C4F3E,
    0x2D3A2F,
    0x1F2A22
  };*/

  static const uint32_t palette[]
      = { 0xFF000000, 0xFF7877EE, 0xFF180719, 0xFFC5421C, 0xFF1D120B,
          0xFF872E47, 0xFF181B0D, 0xFFF1E680, 0xFF111F18, 0xFFF0A28B,
          0xFF0B041E, 0xFF6A57BD, 0xFF1D150E, 0xFF0C8C76, 0xFF0A061D,
          0xFF32904D, 0xFF160018, 0xFF94BCF3, 0xFF042007, 0xFFE7920E,
          0xFF0A0D14, 0xFFB89344, 0xFF0D1C03, 0xFFA9F898, 0xFF040022,
          0xFF3E5330, 0xFF071516, 0xFF9861B8, 0xFF08030C, 0xFFF75CEB,
          0xFF1F2010 };

  /*
  static const uint32_t palette[] = {
    0xa9391f,
    0xa68921,
    0x59601f,
    0x735615,
    0x403f14,
    0xad5621,
    0x312d0c,
    0xf7e847,
    0xe3bb30,
    0x513c0e,
    0x8e7f21,
    0x783411,
    0x742013,
    0xd5b15b,
    0x90963d,
    0x5b4f16,
    0x816f1d,
    0x54210b,
    0x88765f,
    0xccb92e,
    0x140f05,
    0x814c42,
    0xc0a87c,
    0x341f07,
    0x53110b,
    0xf9eb83,
    0x5b4235,
    0x7a5a2e,
    0x170404,
    0x8f5615,
    0x3f2a1e,
    0x512c0b,
    0xc98a19,
    0x64390e,
    0x3f1c07,
    0x241c06,
    0x345e10,
    0x432d2c,
    0x241405,
    0x291f1c,
    0x3e0b06,
    0xbadc46,
    // 0xac446c,
    0x0b1405,
  };
  */

  static const int palette_size = sizeof (palette) / sizeof (palette[0]);

  // static const int samples = 16;

  const int samples = work->samples;

  const double escape_radius_sq = ESCAPE_RADIUS * ESCAPE_RADIUS;

  double scale = work->scale;

  for (int delta_y = 0; delta_y < work->tile; delta_y += work->step)
    {
      int y = work->y + delta_y;

      if (y >= HEIGHT)
        break;

      for (int delta_x = 0; delta_x < work->tile; delta_x += work->step)
        {
          if (work->generation != atomic_load (&g_generation))
            goto clean;

          int x = work->x + delta_x;

          if (x >= WIDTH)
            break;

          double dx = (x - WIDTH / 2.0) * scale;
          double dy = (y - HEIGHT / 2.0) * scale;

          double delta_c_re = dx;
          double delta_c_im = dy;
          double delta_z_re = 0.0;
          double delta_z_im = 0.0;

          // int iter = 0;
          int iter;
          double zn2;

          pthread_mutex_lock (&pixels_done_mutex);
          iter = pixels_done[y * WIDTH + x];
          pthread_mutex_unlock (&pixels_done_mutex);

          if (iter == -1)
            {
              int iter_orbit = 0;

              while (iter < max_iter)
                {
                  double ref_re = work->orbit_re[iter_orbit];
                  double ref_im = work->orbit_im[iter_orbit];

                  double temp_re
                      = 2.0 * (ref_re * delta_z_re - ref_im * delta_z_im);
                  double temp_im
                      = 2.0 * (ref_re * delta_z_im + ref_im * delta_z_re);

                  double dz2_re
                      = delta_z_re * delta_z_re - delta_z_im * delta_z_im;
                  double dz2_im = 2.0 * delta_z_re * delta_z_im;

                  delta_z_re = temp_re + dz2_re + delta_c_re;
                  delta_z_im = temp_im + dz2_im + delta_c_im;

                  if (iter_orbit + 1 > work->orbit_amount - 1)
                    {
                      iter++;
                      continue;
                    }
                  else
                    iter_orbit++;

                  double z_re = work->orbit_re[iter_orbit] + delta_z_re;
                  double z_im = work->orbit_im[iter_orbit] + delta_z_im;

                  if (z_re * z_re + z_im * z_im > escape_radius_sq)
                    break;

                  zn2 = z_re * z_re + z_im * z_im;

                  if ((delta_z_re * delta_z_re + delta_z_im * delta_z_im)
                      > (z_re * z_re + z_im * z_im))
                    {
                      delta_z_re = z_re;
                      delta_z_im = z_im;
                      iter_orbit = 0;
                    }

                  iter++;
                }
            }

          /*
          int iter_orbit = 0;

          while (iter < max_iter)
            {
              double ref_re = work->orbit_re[iter_orbit];
              double ref_im = work->orbit_im[iter_orbit];

              double temp_re
                  = 2.0 * (ref_re * delta_z_re - ref_im * delta_z_im);
              double temp_im
                  = 2.0 * (ref_re * delta_z_im + ref_im * delta_z_re);

              double dz2_re = delta_z_re * delta_z_re - delta_z_im * delta_z_im;
              double dz2_im = 2.0 * delta_z_re * delta_z_im;

              delta_z_re = temp_re + dz2_re + delta_c_re;
              delta_z_im = temp_im + dz2_im + delta_c_im;

              iter_orbit++;

              double z_re = work->orbit_re[iter_orbit] + delta_z_re;
              double z_im = work->orbit_im[iter_orbit] + delta_z_im;

              if (z_re * z_re + z_im * z_im > escape_radius_sq)
                break;

              if ((delta_z_re * delta_z_re + delta_z_im * delta_z_im)
                  > (z_re * z_re + z_im * z_im))
                {
                  delta_z_re = z_re;
                  delta_z_im = z_im;
                  iter_orbit = 0;
                }

              iter++;
            }
          */

          uint32_t color;

          if (iter == max_iter)
            color = 0xFF000000;
          else
            {
              double nu = iter + 1 - log2 (log2 (sqrt (zn2)));

              double freq = 0.1;
              double t = nu * freq;

              t = t - floor (t / palette_size) * palette_size;

              int idx = (int)t;
              double frac = t - idx;

              uint32_t c1 = palette[idx % palette_size];
              uint32_t c2 = palette[(idx + 1) % palette_size];

              color = interpolate_color (c1, c2, frac);
            }

          pthread_mutex_lock (&pixels_mutex);

          for (int step_y = 0; step_y < work->step; ++step_y)
            {
              if (y + step_y >= HEIGHT)
                break;

              for (int step_x = 0; step_x < work->step; ++step_x)
                {
                  if (x + step_x >= WIDTH)
                    break;

                  pixels[(y + step_y) * WIDTH + (x + step_x)] = color;
                }
            }

          pthread_mutex_unlock (&pixels_mutex);
        }
    }

clean:
  free (work);
}

SDL_Texture *
render_text (SDL_Renderer *renderer, TTF_Font *font, const char *text,
             SDL_Color color, int *out_width, int *out_height)
{
  SDL_Surface *surface = TTF_RenderText_Blended (font, text, color);
  SDL_Texture *texture = SDL_CreateTextureFromSurface (renderer, surface);

  *out_width = surface->w;
  *out_height = surface->h;

  SDL_FreeSurface (surface);
  return texture;
}

int
main (void)
{
  srand (time (NULL));
  SDL_Init (SDL_INIT_VIDEO);
  TTF_Init ();

  window = SDL_CreateWindow ("Mandelbrot", SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
  renderer = SDL_CreateRenderer (window, -1, SDL_RENDERER_ACCELERATED);
  texture = SDL_CreateTexture (renderer, SDL_PIXELFORMAT_ARGB8888,
                               SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

  pthread_mutex_init (&pixels_mutex, NULL);
  pthread_mutex_init (&pixels_done_mutex, NULL);

  mpfr_t center_re, center_im, scale;
  mpfr_inits2 (PRECISION_BITS, center_re, center_im, scale, (mpfr_ptr)0);

  // mpfr_set_str (center_re, "-1.985919359960978684453223192193245964271429062666543775386350473746671904875957384480865222226476313620202583817469956970852638701807169521470642552907749357586890444572682637018316051868610025537670443440689026879454018393007724172657727729167322909246742879556044470059151604019800566771833620747885755006452251", 10, MPFR_RNDN);

  // mpfr_set_str (center_im, "-0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000678212430620458622491305267427423408887673261336189549931937020270202016632357548903502696479563088407759416804344221703369528195270240611711018734136689857005888", 10, MPFR_RNDN);

  mpfr_set_d (center_re, -0.75, MPFR_RNDN);
  mpfr_set_d (center_im, 0.00, MPFR_RNDN);
  mpfr_set_d (scale, 0.005, MPFR_RNDN);

  // double scale = 0.005;

  g_orbit_re = calloc (max_iter, sizeof (double));
  g_orbit_im = calloc (max_iter, sizeof (double));

  // double *orbit_re = calloc (max_iter, sizeof (double));
  // double *orbit_im = calloc (max_iter, sizeof (double));

  struct thread_pool *pool;

  pool = thread_pool_create (12, 32768 * 8);

  int redraw = 1;

  TTF_Font *font = TTF_OpenFont ("font.ttf", 24);
  SDL_Color color = { 255, 255, 255, 255 };

  /*SDL_Texture *text_orbit;
  SDL_Rect dst_orbit = { 10, 10, 0, 0 };

  {
    SDL_Surface *surface = TTF_RenderText_Blended (font, "Computing orbit...",
  color); text_orbit = SDL_CreateTextureFromSurface (renderer, surface);
    SDL_FreeSurface (surface);

    SDL_QueryTexture (text_orbit, NULL, NULL, &dst_orbit.w, &dst_orbit.h);
  }*/

  int done = 0;
  int computing_orbit = 0;
  uint32_t start_orbit;

  double zoom_scale = 1.0;
  double zoom_x = 0.0, zoom_y = 0.0;

  uint32_t start, end;

  uint8_t show_information = 0;

  while (1)
    {
      SDL_Event event;

      while (SDL_PollEvent (&event))
        switch (event.type)
          {
          case SDL_QUIT:
            goto quit;
          case SDL_KEYDOWN:
            switch (event.key.keysym.sym)
              {
              case SDLK_LALT:
                show_information = !show_information;
                break;
              case SDLK_PAGEUP:
                atomic_fetch_add (&g_generation, 1);
                thread_pool_clear (pool);
                max_iter *= 2;
                printf ("max_iter=%d\n", max_iter);
                // orbit_re = realloc (orbit_re, max_iter * sizeof (double));
                // orbit_im = realloc (orbit_im, max_iter * sizeof (double));
                g_orbit_re = realloc (g_orbit_re, max_iter * sizeof (double));
                g_orbit_im = realloc (g_orbit_im, max_iter * sizeof (double));
                break;
              case SDLK_PAGEDOWN:
                atomic_fetch_add (&g_generation, 1);
                thread_pool_clear (pool);
                max_iter /= 2;

                if (max_iter <= 64)
                  max_iter = 64;

                printf ("max_iter=%d\n", max_iter);
                // orbit_re = realloc (orbit_re, max_iter * sizeof (double));
                // orbit_im = realloc (orbit_im, max_iter * sizeof (double));
                g_orbit_re = realloc (g_orbit_re, max_iter * sizeof (double));
                g_orbit_im = realloc (g_orbit_im, max_iter * sizeof (double));
                break;
              }
            break;
          case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_RIGHT)
              redraw = 1;
            break;
          case SDL_MOUSEWHEEL:
            {
              int mouse_x, mouse_y;
              SDL_GetMouseState (&mouse_x, &mouse_y);

              double zoom_value = (event.wheel.y > 0) ? 0.75 : 1.25;
              // zoom_scale /= zoom_value;

              double world_x_before = zoom_x + mouse_x / zoom_scale;
              double world_y_before = zoom_y + mouse_y / zoom_scale;

              zoom_scale /= zoom_value;

              double world_x_after = zoom_x + mouse_x / zoom_scale;
              double world_y_after = zoom_y + mouse_y / zoom_scale;

              zoom_x += world_x_before - world_x_after;
              zoom_y += world_y_before - world_y_after;

              mpfr_t old_scale, new_scale;
              mpfr_t re_before, im_before;
              mpfr_t tmp1, tmp2;

              mpfr_inits2 (PRECISION_BITS, old_scale, new_scale, re_before,
                           im_before, tmp1, tmp2, (mpfr_ptr)0);

              mpfr_set (old_scale, scale, MPFR_RNDN);

              mpfr_set_d (tmp1, zoom_value, MPFR_RNDN);
              mpfr_mul (new_scale, scale, tmp1, MPFR_RNDN);

              mpfr_set_d (tmp1, (double)(mouse_x - WIDTH / 2.0), MPFR_RNDN);
              mpfr_mul (tmp2, tmp1, old_scale, MPFR_RNDN);
              mpfr_add (re_before, center_re, tmp2, MPFR_RNDN);

              mpfr_set_d (tmp1, (double)(mouse_y - HEIGHT / 2.0), MPFR_RNDN);
              mpfr_mul (tmp2, tmp1, old_scale, MPFR_RNDN);
              mpfr_add (im_before, center_im, tmp2, MPFR_RNDN);

              mpfr_set_d (tmp1, (double)(mouse_x - WIDTH / 2.0), MPFR_RNDN);
              mpfr_mul (tmp2, tmp1, new_scale, MPFR_RNDN);
              mpfr_sub (center_re, re_before, tmp2, MPFR_RNDN);

              mpfr_set_d (tmp1, (double)(mouse_y - HEIGHT / 2.0), MPFR_RNDN);
              mpfr_mul (tmp2, tmp1, new_scale, MPFR_RNDN);
              mpfr_sub (center_im, im_before, tmp2, MPFR_RNDN);

              mpfr_mul_d (scale, scale, zoom_value, MPFR_RNDN);

              mpfr_clears (old_scale, new_scale, re_before, im_before, tmp1,
                           tmp2, (mpfr_ptr)0);

              redraw = 1;
            }
            break;
          }

      if (redraw)
        {
          done = 0;
          computing_orbit = 1;
          start_orbit = SDL_GetTicks ();
          // printf ("Computing orbit...\n");
          atomic_store (&g_orbit_ready, 0);

          atomic_fetch_add (&g_generation, 1);
          thread_pool_clear (pool);

          struct orbit_work *work;

          work = calloc (1, sizeof (struct orbit_work));
          mpfr_inits2 (PRECISION_BITS, work->center_re, work->center_im,
                       (mpfr_ptr)0);
          mpfr_set (work->center_re, center_re, MPFR_RNDN);
          mpfr_set (work->center_im, center_im, MPFR_RNDN);

          work->scale = mpfr_get_d (scale, MPFR_RNDN);
          work->generation = atomic_load (&g_generation);
          work->orbit_re = calloc (max_iter, sizeof (double));
          work->orbit_im = calloc (max_iter, sizeof (double));

          thread_pool_enqueue (pool, render_compute_orbit_thread, work);

          redraw = 0;

          /*
          start = SDL_GetTicks ();

          uint32_t start_orbit = SDL_GetTicks ();
          render_compute_orbit (center_re, center_im, orbit_re, orbit_im,
          atomic_load (&g_generation)); uint32_t end_orbit = SDL_GetTicks ();
          time_orbit = end_orbit - start_orbit;

          atomic_fetch_add (&g_generation, 1);
          thread_pool_clear (pool);

          for (int step = 64; step >= 1; step /= 2)
            for (int y = 0; y < HEIGHT; y += 64)
              for (int x = 0; x < WIDTH; x += 64)
                {
                  struct render_work *work;
                  work = calloc (1, sizeof (struct render_work));

                  work->x = x;
                  work->y = y;

                  work->tile = 64;
                  work->step = step;

                  work->orbit_re = orbit_re;
                  work->orbit_im = orbit_im;

                  work->scale = mpfr_get_d (scale, MPFR_RNDN);

                  work->generation = atomic_load (&g_generation);

                  thread_pool_enqueue (pool, render_test, work);
                }

          redraw = 0;*/
        }

      // printf ("%d\n", atomic_load (&g_orbit_ready));

      if (atomic_load (&g_orbit_ready))
        {
          for (size_t i = 0; i < WIDTH * HEIGHT; ++i)
            pixels[i] = 0;

          start = SDL_GetTicks ();

          computing_orbit = 0;
          // printf ("Computing image...\n");
          atomic_fetch_add (&g_generation, 1);
          thread_pool_clear (pool);

          for (size_t i = 0; i < WIDTH * HEIGHT; ++i)
            pixels_done[i] = -1;

          const int steps[] = { 16, 4, 1 };
          const int steps_amount = sizeof steps / sizeof (int);

          // for (int step = 64; step != 1; step = 1)
          for (int i = 0; i < steps_amount; ++i)
            {
              int step = steps[i];

              int tile = step;
              if (tile < 8)
                tile = 8;

              for (int y = 0; y < HEIGHT; y += tile)
                for (int x = 0; x < WIDTH; x += tile)
                  {
                    struct render_work *work;
                    work = calloc (1, sizeof (struct render_work));

                    work->x = x;
                    work->y = y;

                    work->tile = tile;
                    work->step = step;
                    work->samples = 1;

                    work->orbit_re = g_orbit_re;
                    work->orbit_im = g_orbit_im;
                    work->orbit_amount = atomic_load (&g_orbit_amount);

                    work->scale = mpfr_get_d (scale, MPFR_RNDN);

                    work->generation = atomic_load (&g_generation);

                    thread_pool_enqueue (pool, render_test, work);
                  }
            }

          atomic_store (&g_orbit_ready, 0);
        }

      if (!done && !computing_orbit
          && thread_pool_get_threads_active (pool) == 0)
        {
          done = 1;

          end = SDL_GetTicks ();

          printf ("%dms %.2e\n", end - start, mpfr_get_d (scale, MPFR_RNDN));

          /*
          for (int y = 0; y < HEIGHT; y += 8)
            for (int x = 0; x < WIDTH; x += 8)
              {
                struct render_work *work;
                work = calloc (1, sizeof (struct render_work));

                work->x = x;
                work->y = y;

                work->tile = 8;
                work->step = 1;
                work->samples = 8;

                work->orbit_re = g_orbit_re;
                work->orbit_im = g_orbit_im;

                work->scale = mpfr_get_d (scale, MPFR_RNDN);

                work->generation = atomic_load (&g_generation);

                thread_pool_enqueue (pool, render_test, work);
              }
              */

          // thread_pool_enqueue (pool, apply_box_blur, NULL);
        }

      // printf ("%d\n", thread_pool_get_threads_active (pool));

      SDL_SetRenderDrawColor (renderer, 66, 61, 57, 255);
      SDL_RenderClear (renderer);
      SDL_UpdateTexture (texture, NULL, pixels, WIDTH * sizeof (uint32_t));

      if (computing_orbit)
        {
        }
      else
        {
          zoom_x = zoom_y = 0;
          zoom_scale = 1.0;
          SDL_RenderCopy (renderer, texture, NULL, NULL);
        }

      SDL_Rect vr = { (-zoom_x) * zoom_scale, (-zoom_y) * zoom_scale,
                      WIDTH * zoom_scale, HEIGHT * zoom_scale };

      SDL_RenderCopy (renderer, texture, NULL, &vr);

      if (computing_orbit && show_information)
        {
          uint32_t end_orbit = SDL_GetTicks ();
          uint32_t time_orbit = end_orbit - start_orbit;

          char buffer[120];
          snprintf (buffer, sizeof buffer, "Computing orbit ... (%dms)",
                    time_orbit);

          SDL_Rect rect = { 20, (HEIGHT - 24) - 20, 0, 0 };
          SDL_Texture *text = render_text (renderer, font, buffer,
                                           (SDL_Color){ 255, 255, 255, 255 },
                                           &rect.w, &rect.h);

          SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
          SDL_RenderFillRect(renderer, &rect);
          SDL_RenderCopy (renderer, text, NULL, &rect);
        }

      if (show_information)
        {
          char buffer[120];
          snprintf (buffer, sizeof buffer, "Zoom: %.2e",
                    mpfr_get_d (scale, MPFR_RNDN));

          SDL_Rect rect = { 20, 20, 0, 0 };
          SDL_Texture *text = render_text (renderer, font, buffer,
                                           (SDL_Color){ 255, 255, 255, 255 },
                                           &rect.w, &rect.h);

          SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
          SDL_RenderFillRect(renderer, &rect);
          SDL_RenderCopy (renderer, text, NULL, &rect);
        }

      SDL_RenderPresent (renderer);

      // SDL_Delay (16);

      // printf ("%dms\n", frame_end - frame_start);
    }

quit:
  SDL_DestroyTexture (texture);
  SDL_DestroyRenderer (renderer);
  SDL_DestroyWindow (window);

  mpfr_clears (center_re, center_im, (mpfr_ptr)0);

  pthread_mutex_destroy (&pixels_mutex);
  pthread_mutex_destroy (&pixels_done_mutex);

  SDL_Quit ();

  return 0;
}

