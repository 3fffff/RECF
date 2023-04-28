export default class DFT {
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

  fft2Core(sig, inverse = false) {
    let signal = sig.constructor === Float32Array ? { re: sig, im: new Float32Array(sig.length) } : sig
    let c_signal = { re: new Float32Array(this.M * this.N), im: new Float32Array(this.M * this.N) };
    let F = { re: new Float32Array(this.M * this.N), im: new Float32Array(this.M * this.N) };
    for (let i = 0; i < this.M; i++) {
      for (let u = 0; u < this.N; u++) {
        let re = 0.0;
        let im = 0.0;
        for (let x = 0; x < this.M; x++) {
          const ix = i + this.M * x;
          const xu = x + this.M * u;
          const d = inverse ? signal.re[xu] : this.wM.re[ix];
          const d1 = inverse ? -signal.im[xu] : this.wM.im[ix];
          const d2 = inverse ? this.Wm.re[ix] : signal.re[xu];
          const d3 = inverse ? this.Wm.im[ix] : signal.im[xu];
          re += d * d2 - d1 * d3;
          im += d * d3 + d1 * d2;
        }
        const x = i + this.M * u;
        c_signal.re[x] = re;
        c_signal.im[x] = inverse ? -im : im;
      }
      for (let u = 0; u < this.N; u++) {
        let re = 0.0;
        let im = 0.0;
        for (let x = 0; x < this.N; x++) {
          const ix = i + this.M * x;
          const xu = x + this.N * u;
          const d = inverse ? c_signal.re[ix] : c_signal.re[ix];
          const d1 = inverse ? c_signal.im[ix] : c_signal.im[ix];
          const d2 = inverse ? this.Wn.re[xu] : this.wN.re[xu];
          const d3 = inverse ? this.Wn.im[xu] : this.wN.im[xu];
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

  fft2(signal) {
    return this.fft2Core(signal, false)
  }

  ifft2(signal) {
    return this.fft2Core(signal, true)
  }

  test() {
    const signal = [0.300194727870250, 0.0677531994661207, 0.473515386787110, 0.125908953423513, 0.929661339629757, 0.0406064330623837, 0.381920034756217, 0.400913925602175,
      0.994591721957246, 0.520070382098481, 0.783811846947656, 0.926042876988079, 0.619331296191692, 0.406657712678292, 0.550653162645015, 0.225701307427070,
      0.290077816430493, 0.232262866366677, 0.637670879874652, 0.319123538768565, 0.907921728122696, 0.248371978659771, 0.539961380395105, 0.101660904565896,
      0.830236446551201, 0.325087199669172, 0.0717264224903104, 0.335998246282916, 0.949771765848046, 0.331372632097893, 0.716832662198039, 0.841137625762162,
      0.721424275969969, 0.721985950565405, 0.320243323475354, 0.628722145136994, 0.930971605311509, 0.608148672695367, 0.842910263724815, 0.993063313559838,
      0.462674830454595, 0.851577150489266, 0.100391170797536, 0.901075248650380, 0.792141939285859, 0.543147459891555, 0.0526616008945997, 0.350792959792590,
      0.176837830445100, 0.532595963561374, 0.692154555234236, 0.666003996935256, 0.0859431916001512, 0.945438568658061, 0.783981384304103, 0.801947597727350]//7*8
    let sim = new Float32Array(signal.length)
    for (let i = 0; i < signal.length; i++)
      sim[i] = 0.0
    const sig = { re: signal, im: sim }
    console.time()
    const res = this.fft2(sig)
    console.timeEnd()
    console.log(res)
    const reso = this.fft2(res, true)
    console.log(reso)
  }
}
