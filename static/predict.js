let imageLoaded = false;
$("#image-selector").change(function () {
  imageLoaded = false;
  let reader = new FileReader();
  reader.onload = function () {
    let dataURL = reader.result;
    $("#selected-image").attr("src", dataURL);
    $("#prediction-list").empty();
    imageLoaded = true;
  };

  let file = $("#image-selector").prop("files")[0];
  //The readAsDataURL method is used to read the contents of the specified Blob or File.
  // When the read operation is finished, the readyState becomes DONE, and the loadend is triggered.
  reader.readAsDataURL(file);
});

let model;
let modelLoaded = false;

$(document).ready(async function () {
  modelLoaded = false;
  $(".progress-bar").show();
  console.log("Loading model...");
  model = await tf.loadGraphModel("model/model.json"); //Load a graph model given a URL to the model definition.
  console.log("Model loaded.");
  $(".progress-bar").hide();
  modelLoaded = true;
});

$("#predict-button").click(async function () {
  if (!modelLoaded) {
    alert("The model must be loaded first");
    return;
  }
  if (!imageLoaded) {
    alert("Please select an image first");
    return;
  }

  let image = $("#selected-image").get(0);

  // Pre-process the image
  console.log("Loading image...");
  let tensor = tf.browser
    .fromPixels(image, 3)
    .resizeNearestNeighbor([224, 224]) // change the image size
    .expandDims()
    .toFloat()
    .reverse(-1); // RGB -> BGR
  let predictions = await model.predict(tensor).data();
  console.log(predictions);
  let top5 = Array.from(predictions)
    .map(function (p, i) {
      // this is Array.map
      return {
        probability: p,
        className: TARGET_CLASSES[i], // we are selecting the value from the obj
      };
    })
    .sort(function (a, b) {
      return b.probability - a.probability;
    })
    .slice(0, 5);

  $("#prediction-list").empty();
  top5.forEach(function (p) {
    $("#prediction-list").append(
      `<li>${p.className}: ${p.probability.toFixed(3) * 100}</li>`,
    );
  });
});
