class DFT {
  constructor(M, N) {
    this.M = M
    this.N = N
    this.wM = this.cosSinTable(M)
    this.wN = this.cosSinTable(N)
    this.Wm = this.cosSinTableInv(M)
    this.Wn = this.cosSinTableInv(N)
  }
  cosSinTable(N) {
    // 1i*(-2*pi) = 0-2*Math.PI*i
    const pi2 = -2 * Math.PI
    const table = { re: new Float32Array(N * N), im: new Float32Array(N * N) }
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        table.re[i * N + j] = /*Math.pow(r,i*j)*/Math.cos((pi2 / N) * i * j)
        table.im[i * N + j] = /*Math.pow(r,i*j)*/Math.sin((pi2 / N) * i * j)
      }
    }
    return table
  }
  cosSinTableInv(N) {
    const pi2 = 2 * Math.PI
    const table = { re: new Float32Array(N * N), im: new Float32Array(N * N) }
    for (let x = 0; x < N; x++) {
      for (let u = 0; u < N; u++) {
        let ai, z_im, z_re, re, brm
        z_im = u * pi2 * x / N;
        z_re = Math.cos(z_im);
        z_im = Math.sin(z_im);
        brm = Math.abs(z_re);
        ai = Math.abs(z_im);
        if (brm > ai) {
          brm = z_im / z_re;
          ai = z_re + brm * z_im;
          re = 1.0 / ai;
          ai = - brm / ai;
        } else if (ai == brm) {
          if (z_re > 0.0) {
            z_re = 0.5;
          } else {
            z_re = -0.5;
          }
          if (z_im > 0.0) {
            ai = 0.5;
          } else {
            ai = -0.5;
          }
          re = z_re / brm;
          ai = - ai / brm;
        } else {
          brm = z_re / z_im;
          ai = z_im + brm * z_re;
          re = brm / ai;
          ai = - 1.0 / ai;
        }
        const i = x + u * N;
        table.re[i] = re / N;
        table.im[i] = ai / N;
      }
    }
    return table
  }
  fft2(signal) {
    let c_signal = { re: new Float32Array(this.M * this.N), im: new Float32Array(this.M * this.N) };
    let F = { re: new Float32Array(this.M * this.N), im: new Float32Array(this.M * this.N) };
    for (let i = 0; i < this.M; i++) {
      for (let u = 0; u < this.N; u++) {
        let re = 0.0;
        let im = 0.0;
        for (let x = 0; x < this.M; x++) {
          const ix = i + this.M * x;
          const xu = x + this.M * u;
          const d = this.wM.re[ix];
          const d1 = this.wM.im[ix];
          const d2 = signal.re[xu];
          const d3 = signal.im[xu];
          re += d * d2 - d1 * d3;
          im += d * d3 + d1 * d2;
        }
        const x = i + this.M * u;
        c_signal.re[x] = re;
        c_signal.im[x] = im;
      }
      for (let u = 0; u < this.N; u++) {
        let re = 0.0;
        let im = 0.0;
        for (let x = 0; x < this.N; x++) {
          const ix = i + this.M * x;
          const xu = x + this.N * u;
          const d = c_signal.re[ix];
          const d1 = c_signal.im[ix];
          const d2 = this.wN.re[xu];
          const d3 = this.wN.im[xu];
          re += d * d2 - d1 * d3;
          im += d * d3 + d1 * d2;
        }
        const x = i + this.M * u;
        F.re[x] = re;
        F.im[x] = im;
      }
    }
    return F
  }
  ifft2(signal) {
    let c_signal = { re: new Float32Array(this.M * this.N), im: new Float32Array(this.M * this.N) };
    let f = { re: new Float32Array(this.N * this.M), im: new Float32Array(this.N * this.M) };
    for (let i = 0; i < this.M; i++) {
      for (let x = 0; x < this.N; x++) {
        let re = 0.0;
        let im = 0.0;
        for (let u = 0; u < this.M; u++) {
          const ux = u + this.M * x;
          const ui = u + this.M * i;
          const d = signal.re[ux];
          const d1 = -signal.im[ux];
          const d2 = this.Wm.re[ui];
          const d3 = this.Wm.im[ui];
          re += d * d2 - d1 * d3;
          im += d * d3 + d1 * d2;
        }
        const u = i + this.M * x;
        c_signal.re[u] = re;
        c_signal.im[u] = -im;
      }
      for (let x = 0; x < this.N; x++) {
        let re = 0.0;
        let im = 0.0;
        for (let u = 0; u < this.N; u++) {
          const iu = i + this.M * u;
          const ux = u + this.N * x;
          const d = c_signal.re[iu];
          const d1 = c_signal.im[iu];
          const d2 = this.Wn.re[ux];
          const d3 = this.Wn.im[ux];
          re += d * d2 - d1 * d3;
          im += d * d3 + d1 * d2;
        }
        const u = i + this.M * x;
        f.re[u] = re;
        f.im[u] = im;
      }
    }
    return f
  }
}