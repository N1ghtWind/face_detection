const video = document.getElementById('videoInput')

// Loading models
//A Promise egy olyan objektum, mely egy aszinkron művelet egy lehetséges végkimenetelét reprezentálja, melyek szintaxisa a következő

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ageGenderNet.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models') //heavier/accurate version of tiny face detector
]).then(start)

function start() {
     //Visszajelzés hogy a modelek betöltöttek
    document.body.append('Models Loaded')
    
    navigator.getUserMedia(
        { video:{} },
        stream => video.srcObject = stream,
        err => console.error(err)
    )
    
    //video.src = '../videos/speech.mp4'
    console.log('video added')
    recognizeFaces()
}

async function recognizeFaces() {

    const labeledDescriptors = await loadLabeledImages()

    // FaceMatcher létrehozása automatikusan hozzárendelt címkékkel
    // a referenciakép észlelési eredményeiből
    // A második paraméter a pontosságot határozza meg, minél kisebb a szám, annál pontosabb a detektálás

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6)
    // console.log(faceMatcher);
    //Lefutatja a függvényt mikor a videó play állapotba kerül
    video.addEventListener('play', async () => {
        console.log('Playing')

        const canvas = faceapi.createCanvasFromMedia(video)

        // Létrehozzunk egy canvas-t
        document.body.append(canvas)

        const displaySize = { width: video.width, height: video.height }
        // Létrehozzuk a displaySize objektumot amelynek a szélessége és hosszusága megegyezik a videó tag méreteivel


        // Beállitjuk a canvas méretét a html videó elem méretére
        // FONTOS: A canvas html tag azután jelenik meg hogy rákattintottunk a play gombra a videón
        faceapi.matchDimensions(canvas, displaySize)
        
        
        //Létreohzzunk egy aszinkron függvényt mely 100ms időközönként fut le.
        setInterval(async () => {

              // Detektáljuk az összes arcot a videóból

            /* withFaceLandmarks():
            Detect all faces in an image + 
            computes 68 Point Face Landmarks for each detected face. 
            */

            /*
            withFaceDescriptors():
            Detect the face with the highest confidence score in an image +
             computes 68 Point Face Landmarks and face descriptor for that face. 
            */
            const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors().withAgeAndGender()
           

            // resize the detected boxes in case your displayed image or video has a different size then the original
            const resizedDetections = faceapi.resizeResults(detections, displaySize)

            
            document.getElementById("curr").innerText = resizedDetections.length;  

            // Minden egyes detektálás után letöröljük a canvas-t, hogy ne maradjonak rajta az előzőleg detektált arcok
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)

            const results = resizedDetections.map((d) => {
                console.log(faceMatcher.findBestMatch(d.descriptor));
                return faceMatcher.findBestMatch(d.descriptor)
            })

            results.forEach( (result, i) => {
              
                const box = resizedDetections[i].detection.box
                const gender = resizedDetections[i].gender;
                let percentage_value = ((100 - (result._distance * 100)) * 1.5)

                if(percentage_value >= 100) {
                    percentage_value = 100;
                }

                result._distance = percentage_value.toFixed(2).toString() + "%" ;

                const drawBox = new faceapi.draw.DrawBox(box, { label: result._label + " | " + result._distance + " | " + gender })
               
                drawBox.draw(canvas)
            })
        }, 100)


        
    })
}


function loadLabeledImages() {
     //const labels = ['Black Widow', 'Captain America', 'Hawkeye' , 'Jim Rhodes', 'Tony Stark', 'Thor', 'Captain Marvel']
    
    
    // Létrehozzuk a labels tömböt

    const labels = ['Krisztian','Jozsi'] // for WebCam
    return Promise.all(

        // Aszinkron függvényel végig iterálunk a label tömbön, hasonló mint egy foreach
        labels.map(async (label)=>{
            // Létrehozzuk a descriptions tömböt
            const descriptions = []
            // Itt egy for ciklust használunk ami 2-ig megy, mivel minden egyes személynek a mappájában 2 kép kell hogy legyen, ha több lesz, akkor array out of range hiba lép fel
            for(let i=1; i<=3; i++) {

                //ez betölti az adott képet a faceapi detektáláshoz
                const img = await faceapi.fetchImage(`../labeled_images/${label}/${i}.jpg`)
                // ezzel a parancsal az adott képen melyet betöltöttünk, detektáljuk az arcot a képen

                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                console.log(label + i + JSON.stringify(detections))
                //A descriptions tömbbe belerakjuk az észlelt arcnak az arcleírását, ez egy hosszú string, amely alapján a faceapi képes összehasonlitani különböző arcokat

                descriptions.push(detections.descriptor)
            }
            document.body.append(label+' Faces Loaded | ')
            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}