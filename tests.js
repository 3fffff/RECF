import ReCF from './ReCF_real.js'

export function mulTest() {
  const cf = new ReCF()
  const testArr = new Float32Array(2500)
  for (let i = 0; i < 2500; i++)
    testArr[i] = i
  const result = cf.mul(testArr, 20)
  console.log(result)
}

export function addTest() {
  const cf = new ReCF()
  const testArr = new Float32Array(2500)
  for (let i = 0; i < 2500; i++)
    testArr[i] = i
  const result = cf.add(testArr, 20)
  console.log(result)
}

export function subTest() {
  const cf = new ReCF()
  const testArr = new Float32Array(2500)
  for (let i = 0; i < 2500; i++)
    testArr[i] = i
  const result = cf.sub(testArr, 20)
  console.log(result)
}

export function channelMultiplyTest() {
  const cf = new ReCF()
  let testA = []
  let testB = []
  for (let j = 0; j < 3; j++) {
    let A =new Float32Array(2500)
    let B =new Float32Array(2500)
    for (let i = 0; i < 2500; i++) {
      A[i] = i
      B[i] = i
    }
    testA.push(A)
    testB.push(B)
  }
  const result = cf.channelMultiply(testA, testB)
  console.log(result)
}

export function extract_tracked_region_spec_test(){
  
}
