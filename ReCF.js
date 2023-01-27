import DFT from './dft.js'

export default class ReCF {
  constructor() {
    this.model_sz = new Uint32Array([50, 50]);
    this.target_padding = 2;
    this.update_rate = 0.14;
    this.sigma_factor = 1 / 16;
    this.admm = 2
    this.init = false
    this.channels = 3
  }

  preprocess(data, div, sub, region, channels_in) {
    let result = new Float32Array(this.channels * region.w * region.h), pos = 0
    for (let y = region.y; y < region.y + region.h; y++) {
      for (let x = region.x; x < region.x + region.w; x++) {
        const pp = (y * data.width + x) * channels_in;
        for (let i = 0; i < this.channels; i++)
          result[pos++] = (data.data[pp + i] * div + sub)
      }
    }
    return result;
  }

  mul(matrix, number) {
    let result = new Float32Array(matrix.length)
    for (let i = 0; i < matrix.length; i++)
      result[i] = matrix[i] * number
    return result;
  }

  multiply(mat1, mat2) {
    let result = new Float32Array(mat1.length)
    for (let i = 0; i < mat1.length; i++)
      result[i] = mat1[i] * mat2[i]
    return result
  }

  add(matrix, number) {
    let result = new Float32Array(matrix.length)
    for (let i = 0; i < matrix.length; i++)
      result[i] = matrix[i] + number
    return result;
  }

  addition(mat1, mat2) {
    let result = new Float32Array(mat1.length)
    for (let i = 0; i < mat1.length; i++)
      result[i] = mat1[i] + mat2[i]
    return result;
  }

  sub(matrix, number) {
    let result = new Float32Array(matrix.length)
    for (let i = 0; i < matrix.length; i++)
      result[i] = matrix[i] - number
    return result;
  }

  subtraction(mat1, mat2) {
    let result = new Float32Array(mat1.length)
    for (let i = 0; i < mat1.length; i++)
      result[i] = mat1[i] - mat2[i]
    return result;
  }

  div(matrix, number) {
    let result = Float32Array(matrix.length)
    for (let i = 0; i < matrix.length; i++)
      result[i] = matrix[i] / number
    return result;
  }

  divide(mat1, mat2) {
    let result = new Float32Array(mat1.length)
    for (let i = 0; i < mat1.length; i++)
      result[i] = mat1[i] / mat2[i]
    return result;
  }

  channel_multiply(a, b) {
    let sum = new Float32Array(a[0].length);
    for (let i = 0; i < a.length; ++i) {
      const prod = this.multiply(a[i], b[i]);
      sum = this.addition(sum, prod)
    }
    return sum;
  }

  minMaxLoc(array) {
    let max = -999999, maxpos = { x: 0, y: 0 };
    for (let x = 0; x < this.model_sz[0]; x++) {
      for (let y = 0; y < this.model_sz[1]; y++) {
        const val = array[(y * this.model_sz[0]) + x];
        if (max < val) {
          max = val;
          maxpos.x = x;
          maxpos.y = y;
        }
      }
    }
    return maxpos;
  }

  extract_tracked_region(image, region) {
    const extractionRegion = {
      x: region.x - ~~(region.w / 2),
      y: region.y - ~~(region.h / 2),
      w: region.w,
      h: region.h
    };

    //make sure the patch is not completely outside the image
    if (extractionRegion.x < 0)
      extractionRegion.x = 0
    if (extractionRegion.y < 0)
      extractionRegion.y = 0
    if (extractionRegion.x > image.width)
      extractionRegion.x = image.width
    if (extractionRegion.y > image.height)
      extractionRegion.y = image.height

    const extract_region = this.preprocess(image, 1 / 255, -0.5, extractionRegion, this.channels);
    return this.resize(extract_region, { width: extractionRegion.w, height: extractionRegion.h }, { width: this.model_sz[0], height: this.model_sz[1] });
  }

  resize(input, src, dst, channels = 3) {
    const red = new Float32Array(dst.height * dst.width)
    const green = new Float32Array(dst.height * dst.width)
    const blue = new Float32Array(dst.height * dst.width)
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
        red[(y * dst.width + x)] =
          dx2 * dy2 * input[x11 + 0] + dx1 * dy2 * input[x21 + 0] + dx2 * dy1 * input[x12 + 0] + dx1 * dy1 * input[x22 + 0];
        green[(y * dst.width + x)] =
          dx2 * dy2 * input[x11 + 1] + dx1 * dy2 * input[x21 + 1] + dx2 * dy1 * input[x12 + 1] + dx1 * dy1 * input[x22 + 1];
        blue[(y * dst.width + x)] =
          dx2 * dy2 * input[x11 + 2] + dx1 * dy2 * input[x21 + 2] + dx2 * dy1 * input[x12 + 2] + dx1 * dy1 * input[x22 + 2];
      }
    }
    return [red, green, blue]
  }

  track(image) {
    this.detect(image);
    this.boundingBox = { x: this.target.x, y: this.target.y, w: ~~(this.target.w / this.target_padding), h: ~~(this.target.h / this.target_padding) }
    this.update(image);
  }

  initialize(image, region) {
    this.target = { x: region.x + ~~(region.w / 2), y: region.y + ~~(region.h / 2), w: this.target_padding * region.w, h: this.target_padding * region.h }
    this.dft = new DFT(this.model_sz[0], this.model_sz[1])
    this.scale_factor = Math.sqrt(this.target.w * this.target.h / (this.model_sz[0] * this.model_sz[1]));
    const resize_factor = (1 / this.scale_factor) * (1 / this.target_padding);
    this.model_xf = this.create_array()
    this.labelsf = this.make_labels({ w: ~~(this.target.w * resize_factor), h: ~~(this.target.h * resize_factor) }, this.model_sz);
    this.filterf = this.create_array()
    this.window = this.hann_window();
    this.update(image);
    this.boundingBox = { x: this.target.x, y: this.target.y, w: ~~(this.target.w / this.target_padding), h: ~~(this.target.h / this.target_padding) }
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
    const feature_data = this.extract_tracked_region(patch, this.target);
    return this.fft2(feature_data);
  }

  detect(image) {
    const response = this.compute_response(image);
    const maxpos = this.minMaxLoc(response);
    this.target.x += Math.round(this.shift_index(maxpos.x, this.model_sz[0]) * this.scale_factor / 2);
    this.target.y += Math.round(this.shift_index(maxpos.y, this.model_sz[1]) * this.scale_factor / 2)
    console.log(response)
    console.log(maxpos)
  }

  compute_response(image) {
    const feature_vecf = this.compute_feature_vec(image);
    const resp_dft = this.channel_multiply(feature_vecf, this.filterf);
    return this.dft.ifft2(resp_dft);
  }

  fft2(channels) {
    const channelsf = Array(channels.length)
    for (let i = 0; i < channels.length; ++i) {
      const windowed = this.multiply(channels[i], this.window);
      channelsf[i] = this.dft.fft2(windowed);
    }
    return channelsf;
  }

  shift_index(index, length) {
    return (index > length / 2) ? -length + index : index;
  }

  hann_window() {
    const array = new Float32Array(this.model_sz[0] * this.model_sz[1])
    const tvec = Array(this.model_sz[0] + this.target_padding)
    for (let i = 0; i < Math.round((this.model_sz[0] + this.target_padding) / 2); i++) {
      const hann = 0.5 - 0.5 * Math.cos(2 * Math.PI * (i / (this.model_sz[0] + 1)));
      tvec[i] = hann
      tvec[this.model_sz[0] + 1 - i] = hann
    }
    for (let k = 1; k < this.model_sz[0] + 1; ++k)
      for (let j = 1; j < this.model_sz[0] + 1; ++j)
        array[(k - 1) * this.model_sz[0] + (j - 1)] = tvec[k] * tvec[j];
    return array
  }

  make_labels(target_size, model_sz) {
    const labels = new Float32Array(model_sz[1] * model_sz[0])
    const sigma = Math.sqrt(target_size.h * target_size.w) * this.sigma_factor;
    const constant = -0.5 / Math.pow(sigma, 2);
    for (let x = 0; x < this.model_sz[0]; x++) {
      for (let y = 0; y < this.model_sz[1]; y++) {
        const shift_x = this.shift_index(x, this.model_sz[0]);
        const shift_y = this.shift_index(y, this.model_sz[1]);
        const value = Math.exp(constant * (Math.pow(shift_x, 2) + Math.pow(shift_y, 2)));
        labels[y * this.model_sz[0] + x] = value;
      }
    }
    return this.dft.fft2(labels);
  }

  create_array() {
    const res = []
    for (let i = 0; i < this.channels; i++)res.push(new Float32Array(this.model_sz[1] * this.model_sz[0]))
    return res
  }

  compute_ADMM() {
    let l_f = this.create_array();
    let h_f = this.create_array();
    this.filterf = this.create_array()
    let mu = 1;
    const T = this.model_sz[0] * this.model_sz[1];
    const S_xx = this.channel_multiply(this.model_xf, this.model_xf);
    for (let i = 0; i < this.admm; i++) {
      const B = this.add(S_xx, T * mu);
      const S_lx = this.channel_multiply(l_f, this.model_xf);
      const S_hx = this.channel_multiply(h_f, this.model_xf);
      for (let j = 0; j < this.model_xf.length; j++) {
        const mlabelf = this.multiply(this.labelsf, this.model_xf[j]);
        const S_xxyf = this.multiply(S_xx, mlabelf);
        const mS_lx = this.multiply(S_lx, this.model_xf[j]);
        const mS_hx = this.multiply(S_hx, this.model_xf[j]);
        const BS = this.divide(this.addition(this.subtraction(this.mul(S_xxyf, 1 / (T * mu)), this.mul(mS_lx, 1 / mu)), mS_hx), B);
        this.filterf[j] = this.subtraction(this.addition(this.subtraction(this.mul(mlabelf, 1 / (T * mu)), this.mul(l_f[j], 1 / mu)), h_f[j]), BS);
        const h = this.dft.ifft2(this.addition(this.mul(this.filterf[j], mu), l_f[j]));
        const t = this.extract_tracked_region_spec(this.mul(h, (1 / mu)), this.model_sz);
        h_f[j] = this.dft.fft2(t);
        l_f[j] = this.addition(l_f[j], this.subtraction(this.mul(this.filterf[j], mu), h_f[j]));
      }
      mu *= 10;
    }
  }

  extract_tracked_region_spec(model, output_sz) {
    const lp = new Float32Array(output_sz[1] * output_sz[0])
    const crop_width = Math.ceil(output_sz.width / 4)
    const crop_height = Math.ceil(output_sz.height / 4)
    for (let x = 0; x < output_sz.width; x++) {
      for (let y = 0; y < output_sz.height; y++) {
        if ((x > crop_width + 1 && x < crop_width + 1 + output_sz.width / 2) && crop_height + 1 && y < crop_height + 1 + output_sz.height / 2)
          lp[y * output_sz[0] + x] = model[y * output_sz[0] + x];
        else
          lp[y * output_sz[0] + x] = 0;
      }
    }
    return lp;
  }
}
