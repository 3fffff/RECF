class ReCF {
  constructor() {
    this.model_sz = [50, 50];
    this.target_padding = 2.0;
    this.update_rate = 0.12;
    this.sigma_factor = 1.0 / 16.0;
    this.admm = 2
    this.init = false
  }

  preprocess(data, div, sub, region) {
    // in-place
    let result = new Float32Array(3 * region.w * region.h), pos = 0
    for (let y = region.y; y < region.y + region.h; y++) {
      for (let x = region.x; x < region.x + region.w; x++) {
        const pp = (y * data.width + x) * 4;
        result[pos++] = (data.data[pp + 0] * div + sub)
        result[pos++] = (data.data[pp + 1] * div + sub);
        result[pos++] = (data.data[pp + 2] * div + sub);
      }
    }
    return result;
  }

  conj(matrix) {
    let result = { re: new Float32Array(matrix.re.length), im: new Float32Array(matrix.re.length) };
    for (let i = 0; i < matrix.re.length; i++) {
      result.re[i] = matrix.re[i]
      result.im[i] = -matrix.im[i];
    }
    return result;
  }
  mul(matrix, number) {
    let result = { re: new Float32Array(matrix.re.length), im: new Float32Array(matrix.re.length) };
    for (let r = 0; r < matrix.re.length; r++) {
      result.re[r] = matrix.re[r] * number
      result.im[r] = matrix.im[r] * number
    }
    return result;
  }
  multiply(cn1, cn2) {
    let result = { re: new Float32Array(cn1.re.length), im: new Float32Array(cn1.re.length) };
    for (let r = 0; r < cn1.re.length; r++) {
      result.re[r] = (cn1.re[r] * cn2.re[r]) - (cn1.im[r] * cn2.im[r]);
      result.im[r] = (cn1.re[r] * cn2.im[r]) + (cn1.im[r] * cn2.re[r]);
    }
    return result;
  }
  add(matrix, number) {
    let result = { re: new Float32Array(matrix.re.length), im: new Float32Array(matrix.re.length) };
    for (let r = 0; r < matrix.re.length; r++) {
      result.re[r] = matrix.re[r] + number
      result.im[r] = matrix.im[r] + number
    }
    return result;
  }
  addition(mat1, mat2) {
    let result = { re: new Float32Array(mat1.re.length), im: new Float32Array(mat1.re.length) };
    for (let r = 0; r < mat1.re.length; r++) {
      result.re[r] = mat1.re[r] + mat2.re[r]
      result.im[r] = mat1.im[r] + mat2.im[r]
    }
    return result;
  }
  sub(matrix, number) {
    let result = { re: new Float32Array(matrix.re.length), im: new Float32Array(matrix.re.length) };
    for (let r = 0; r < matrix.re.length; r++) {
      result.re[r] = matrix.re[r] - number
      result.im[r] = matrix.im[r] - number
    }
    return result;
  }
  subtraction(mat1, mat2) {
    let result = { re: new Float32Array(mat1.re.length), im: new Float32Array(mat1.re.length) };
    for (let r = 0; r < mat1.re.length; r++) {
      result.re[r] = mat1.re[r] - mat2.re[r]
      result.im[r] = mat1.im[r] - mat2.im[r]
    }
    return result;
  }
  div(matrix, number) {
    let result = { re: new Float32Array(matrix.re.length), im: new Float32Array(matrix.re.length) };
    for (let r = 0; r < matrix.re.length; r++) {
      result.re[r] = matrix.re[r] / number
      result.im[r] = matrix.im[r] / number
    }
    return result;
  }
  divide(cn1, cn2) {
    let result = { re: new Float32Array(cn1.re.length), im: new Float32Array(cn1.re.length) };
    for (let r = 0; r < cn1.re.length; r++) {
      result.re[r] = ((cn1.re[r] * cn2.re[r]) + (cn1.im[r] * cn2.im[r])) / ((cn2.re[r] * cn2.re[r]) + (cn2.im[r] * cn2.im[r]));
      result.im[r] = ((cn1.im[r] * cn2.re[r]) - (cn1.re[r] * cn2.im[r])) / ((cn2.re[r] * cn2.re[r]) + (cn2.im[r] * cn2.im[r]));
    }
    return result;
  }
  channelMultiply(a, b, conj) {
    const sum = [];
    if (b instanceof Array)
      for (let i = 0; i < a.length; ++i)sum[i] = this.multiply(a[i], conj ? this.conj(b[i]) : b[i]);
    else for (let i = 0; i < a.length; ++i)sum[i] = this.multiply(a[i], conj ? this.conj(b) : b);
    return sum;
  }

  channelSum(a) {
    let sum = { re: new Float32Array(a[0].re.length), im: new Float32Array(a[0].re.length) };
    for (let i = 0; i < a.length; ++i) {
      sum = this.addition(sum, a[i]);
      console.log(sum)
    }
    return sum;
  }
  minMaxLoc(array) {
    let max = 0, maxpos = { x: 0, y: 0 };
    for (let x = 0; x < this.model_sz[0]; x++) {
      for (let y = 0; y < this.model_sz[1]; y++) {
        const val = array[(y * this.model_sz[0]) + x];
        if (max < val) {
          max = val;
          maxpos.x = y
          maxpos.y = y;
        }
      }
    }
    console.log(maxpos)
    return maxpos;
  }
  extractTrackedRegion(image, region) {
    const extractionRegion = {
      x: region.x - region.w / 2.0,
      y: region.y - region.h / 2.0,
      w: region.w,
      h: region.h
    };

    //make sure the patch is not completely outside the image
    if (extractionRegion.x + extractionRegion.w > 0 && extractionRegion.y + extractionRegion.h > 0 &&
      extractionRegion.x < image.width && extractionRegion.y < image.height) {
      const real_patch = this.preprocess(image, 1 / 255, -0.5, extractionRegion);
      return this.resize(real_patch, { width: extractionRegion.w, height: extractionRegion.h }, { width: this.model_sz[0], height: this.model_sz[1] });
    } else throw new Error("no extract region")
  }

  resize(input, src, dst, channels = 3) {
    const red = { re: new Float32Array(dst.height * dst.width), im: new Float32Array(dst.height * dst.width) };
    const green = { re: new Float32Array(dst.height * dst.width), im: new Float32Array(dst.height * dst.width) };
    const blue = { re: new Float32Array(dst.height * dst.width), im: new Float32Array(dst.height * dst.width) };
    const scale = [(dst.width / src.width), (dst.height / src.height)]
    for (let y = 0; y < dst.height; ++y) {
      const inY1 = Math.min(~~(y / scale[1]), src.height - 1);
      const inY2 = Math.min(inY1 + 1, src.height - 1);
      const dy1 = ~~(y / scale[1]) - inY1, dy2 = ~~(y / scale[1]) - inY2;
      for (let x = 0; x < dst.width; ++x) {
        const inX1 = Math.min(~~(x / (dst.width / src.width)), src.width - 1);
        const inX2 = Math.min(inX1 + 1, src.width - 1);
        const dx1 = ~~(x / (scale[0])) - inX1, dx2 = ~~(x / (scale[0])) - inX2
        const x11 = (src.width * inY1 + inX1) * channels;
        const x21 = (src.width * inY1 + inX2) * channels;
        const x12 = (src.width * inY2 + inX1) * channels;
        const x22 = (src.width * inY2 + inX2) * channels;
        red.re[(y * dst.width + x)] =
          dx2 * dy2 * input[x11 + 0] + dx1 * dy2 * input[x21 + 0] + dx2 * dy1 * input[x12 + 0] + dx1 * dy1 * input[x22 + 0];
        green.re[(y * dst.width + x)] =
          dx2 * dy2 * input[x11 + 1] + dx1 * dy2 * input[x21 + 1] + dx2 * dy1 * input[x12 + 1] + dx1 * dy1 * input[x22 + 1];
        blue.re[(y * dst.width + x)] =
          dx2 * dy2 * input[x11 + 2] + dx1 * dy2 * input[x21 + 2] + dx2 * dy1 * input[x12 + 2] + dx1 * dy1 * input[x22 + 2];
      }
    }
    return [red, green, blue]
  }

  updateImpl(image) {
    this.detect(image);
    this.targetOut = { x: this.target.x, y: this.target.y, w: this.target.w / this.target_padding, h: this.target.h / this.target_padding }
    this.update(image);
  }

  initialize(image, region) {
    this.target = { x: region.x + ~~(region.w / 2), y: region.y + ~~(region.h / 2), w: this.target_padding * region.w, h: this.target_padding * region.h }
    this.dft = new DFT(this.model_sz[0], this.model_sz[1])
    this.scale_factor = Math.sqrt(this.target.w * this.target.h / (this.model_sz[0] * this.model_sz[1]));
    const resize_factor = (1.0 / this.scale_factor) * (1.0 / this.target_padding);
    this.model_xf = this.init_array()
    this.labelsf = this.make_labels({ w: ~~(this.target.w * resize_factor), h: ~~(this.target.h * resize_factor) }, this.model_sz);
    this.filterf = this.init_array()
    this.window = this.hann_window();
    this.update(image);
    this.targetOut = { x: this.target.x, y: this.target.y, w: this.target.w / this.target_padding, h: this.target.h / this.target_padding }
    this.init = true
  }

  update(image) {
    const feature_vecf = this.compute_feature_vec(image);
    if (!this.init) this.model_xf = feature_vecf;
    else {
      for (let i = 0; i < this.model_xf.length; i++)
        this.model_xf[i] = this.addition(this.mul(this.model_xf[i], 1 - this.update_rate), this.mul(feature_vecf[i], this.update_rate));
    }
    this.compute_ADMM();
  }

  compute_feature_vec(patch) {
    const feature_data = this.extractTrackedRegion(patch, this.target);
    return this.fft2(feature_data);
  }

  detect(image) {
    const response = this.compute_response(image);
    console.log(response)
    const maxpos = this.minMaxLoc(response.re);
    //region w or h
    this.target.x = this.target.x + Math.round(this.shift_index(maxpos.x, this.model_sz[0]) * this.scale_factor / 2);
    this.target.y = this.target.y + Math.round(this.shift_index(maxpos.y, this.model_sz[1]) * this.scale_factor / 2)
    console.log(this.target)
  }

  compute_response(image) {
    const feature_vecf = this.compute_feature_vec(image);
    const resp_dft = this.channelSum(this.channelMultiply(feature_vecf, this.filterf, true));
    return this.dft.ifft2(resp_dft);
  }

  fft2(channels) {
    const channelsf = Array(channels.length)
    for (let i = 0; i < channels.length; ++i) {
      const windowed = this.multiply(channels[i], this.window);
      channelsf[i] = this.dft.fft2(windowed);
      //console.log(channels[i])
      //console.log(channelsf[i])
    }
    return channelsf;
  }

  shift_index(index, length) {
    return (index > length / 2) ? -length + index : index;
  }
  hann_window() {
    const array = { re: new Float32Array(this.model_sz[0] * this.model_sz[1]), im: new Float32Array(this.model_sz[0] * this.model_sz[1]) }
    const tvec = Array(this.model_sz[0] + this.target_padding)
    for (let i = 0; i < Math.round((this.model_sz[0] + this.target_padding) / 2); i++) {
      const hann = 0.5 - 0.5 * Math.cos(2 * Math.PI * (i / (this.model_sz[0] + 1)));
      tvec[i] = hann
      tvec[this.model_sz[0] + 1 - i] = hann
    }
    for (let k = 1; k < this.model_sz[0] + 1; ++k)
      for (let j = 1; j < this.model_sz[0] + 1; ++j)
        array.re[(k - 1) * this.model_sz[0] + (j - 1)] = tvec[k] * tvec[j];
    return array
  }
  make_labels(target_size, model_sz) {
    const labels = { re: new Float32Array(model_sz[1] * model_sz[0]), im: new Float32Array(model_sz[1] * model_sz[0]) };
    const sigma = Math.sqrt(target_size.h * target_size.w) * this.sigma_factor;
    const constant = -0.5 / Math.pow(sigma, 2);
    for (let x = 0; x < this.model_sz[0]; x++) {
      for (let y = 0; y < this.model_sz[1]; y++) {
        const shift_x = this.shift_index(x, this.model_sz[0]);
        const shift_y = this.shift_index(y, this.model_sz[1]);
        const value = Math.exp(constant * (Math.pow(shift_x, 2) + Math.pow(shift_y, 2)));
        labels.re[y * this.model_sz[0] + x] = value;
      }
    }
    return this.dft.fft2(labels);
  }
  init_array() {
    const res = []
    for (let i = 0; i < 3; i++)res.push({ re: new Float32Array(this.model_sz[1] * this.model_sz[0]), im: new Float32Array(this.model_sz[1] * this.model_sz[0]) })
    return res
  }
  compute_ADMM() {
    let l_f = this.init_array();
    let h_f = this.init_array();
    let mu = 1;
    const T = this.model_sz[0] * this.model_sz[1];
    const S_xx = this.channelSum(this.channelMultiply(this.model_xf, this.model_xf, false));
    this.filterf = this.init_array()
    for (let i = 0; i < this.admm; i++) {
      const B = this.add(S_xx, T * mu);
      const S_lx = this.channelSum(this.channelMultiply(l_f, this.model_xf, false));
      const S_hx = this.channelSum(this.channelMultiply(h_f, this.model_xf, false));
      for (let j = 0; j < this.model_xf.length; j++) {
        const mlabelf = this.multiply(this.labelsf, this.model_xf[j]);
        const S_xxyf = this.multiply(S_xx, mlabelf);
        const mS_lx = this.multiply(S_lx, this.model_xf[j]);
        const mS_hx = this.multiply(S_hx, this.model_xf[j]);
        const ghj = this.divide(this.addition(this.subtraction(this.mul(S_xxyf, 1 / (T * mu)), this.mul(mS_lx, 1 / mu)), mS_hx), B);
        this.filterf[j] = this.subtraction(this.addition(this.subtraction(this.mul(mlabelf, 1 / (T * mu)), this.mul(l_f[j], 1 / mu)), h_f[j]), ghj);
        const h = this.dft.ifft2(this.addition(this.mul(this.filterf[j], mu), l_f[j]));
        const t = this.extractTrackedRegionSpec(this.mul(h, (1 / mu)), this.model_sz);
        h_f[j] = this.dft.fft2(t);
        l_f[j] = this.addition(l_f[j], this.subtraction(this.mul(this.filterf[j], mu), h_f[j]));
      }
      mu *= 10;
    }
  }

  extractTrackedRegionSpec(model, output_sz) {
    const lp = { re: new Float32Array(this.model_sz[1] * this.model_sz[0]), im: new Float32Array(this.model_sz[1] * this.model_sz[0]) };
    for (let x = 0; x < output_sz.w; x++) {
      for (let y = 0; y < output_sz.h; y++) {
        if ((x < output_sz.w / 4 || x > output_sz.w / 2) && (y < output_sz.h / 4 || y > output_sz.h / 2))
          lp.re[y * this.output_sz[0] + x] = 0;
        else lp.re[y * this.output_sz[0] + x] = model.re[y * this.output_sz[0] + x];
      }
    }
    return lp;
  }
}
