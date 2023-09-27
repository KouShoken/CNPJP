function getBoxesData_path(imageUrl, callback) {
	const endpoint = "/cars/mark";

	// Get file from file url
	$.ajax({
		beforeSend: function () {
			$("#ajax-loadingIcon").removeClass("d-none");
		},
		url: imageUrl,
		type: 'GET',
		xhrFields: {
			responseType: 'blob' // Blob
		},
		success: function (blob) {
			// 第二步：创建 FormData 并添加 Blob
			const formData = new FormData();
			const fileName = `${uuidv4()}.jpg`
			formData.append('file', blob, fileName);

			// 第三步：发送 FormData 到服务器
			$.ajax({
				beforeSend: function () {
					$("#ajax-loadingIcon").removeClass("d-none");
				},
				url: endpoint,
				type: 'POST',
				data: formData,
				processData: false,
				contentType: false,
				success: function (response) {
					// console.debug(response);
					callback(true, response);
				},
				error: function (errMsg) {
					console.error("Failed to get mark data：", errMsg);
					callback(false, errMsg);
				},
				complete: function () {
					$("#ajax-loadingIcon").addClass("d-none");
				}
			});
		},
		error: function (errMsg) {
			console.error(`Failed to pull image file： ${imageUrl}`, errMsg);
			callback(false, errMsg);
		},
		complete: function () {
			// W: Canceling this will cause the loading icon to not display properly
			// $("#ajax-loadingIcon").addClass("d-none");
		}
	});
}

function drawBoxes(data, container, img) {
	if (data === undefined)
		return;

	const imgWidth = img.width();
	const imgHeight = img.height();
	const boxList = data.data;

	// 遍历数据并为每个对象创建一个框
	$.each(boxList, function (index, item) {
		if (item.box && item.box.x && item.box.y) {
			const box = item.box;
			const x = box.x;
			const y = box.y;

			// 计算框的位置和大小
			var left = x[0] * imgWidth;
			var top = y[0] * imgHeight;
			var width = (x[1] - x[0]) * imgWidth;
			var height = (y[1] - y[0]) * imgHeight;

			// 创建框并添加到容器中
			const $box = $(`<div data-detial='${JSON.stringify(item)}' class="markBox"></div>`);
			$box.css({
				position: 'absolute',
				left: left + 'px',
				top: top + 'px',
				width: width + 'px',
				height: height + 'px',
				border: '1px solid #FF00FF', // 您可以根据需要更改边框颜色
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
