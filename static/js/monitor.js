function getBoxesData_urlPath(imageUrl, callback) {
    /**
     * image data from URL Path
     */

    $.ajax({
        beforeSend: function () {
            $("#ajax-loadingIcon").removeClass("d-none");
        },
        url: imageUrl,
        type: 'GET',
        xhrFields: {
            responseType: 'blob' // 指定返回类型为 Blob
        },
        success: function (blob) {
            getCarsPosition(blob)
                .then(result => {
                    console.log(result);
                    callback(result.success, result.data)
                })
                .catch(error => {
                    console.error(error);
                })
                .finally(() => {
                    // Hide loading dot
                    $("#ajax-loadingIcon").addClass("d-none");
                });
        },
        error: function (errMsg) {
            console.error(`Failed to pull image file： ${imageUrl}`, errMsg);
            callback(false, errMsg); // When fail
        }
    });
}

function drawBoxes(data, container, img) {
    /**
     * drawBoxes on Web
     */
    if (data === undefined)
        return;

    const imgWidth = img.width();
    const imgHeight = img.height();
    const boxList = data.data;

    // Draw boxes
    $.each(boxList, function (index, item) {
        if (item.box && item.box.x && item.box.y) {
            const box = item.box;
            const x = box.x;
            const y = box.y;

            // Pos and Size
            var left = x[0] * imgWidth;
            var top = y[0] * imgHeight;
            var width = (x[1] - x[0]) * imgWidth;
            var height = (y[1] - y[0]) * imgHeight;

            // Make box objects
            const $box = $(`<div data-detial='${JSON.stringify(item)}' class="markBox"></div>`);
            $box.css({
                position: 'absolute',
                left: left + 'px',
                top: top + 'px',
                width: width + 'px',
                height: height + 'px',
                border: '1px solid #FF00FF', // boxes color
                boxSizing: 'border-box',
                color: '#FF00FF'
            });

            container.append($box);
        } else {
            console.error('Invalid data at index ' + index, item);
        }
        return container;
    });
}

function clearMarkedBoxes() {
    $(".markBox").remove();
}
